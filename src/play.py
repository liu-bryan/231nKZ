"""Live play loop: capture game window -> detector -> policy -> inject inputs.

Wires screen capture (mss), Runner (one detector + that pipeline's policy), and pynput.

Use exactly one pipeline per run (--pipeline yolo or --pipeline rtdetr).
Object lists from YOLO and RT-DETR are never merged.

Requirements
------------
- Trained detector + policy for the chosen pipeline (or --detect-only)
- Game window visible; Katana ZERO should be focused for inputs to land in-game
- **macOS:** Screen Recording (mss capture) + Accessibility (pynput keyboard/mouse).
  Optional ``--window-title`` uses Quartz to find the game window.
- **Windows / Linux:** ``mss`` and ``pynput`` work without extra permissions; pass
  ``--rect left,top,width,height`` (auto window detect is macOS-only today).

Usage
-----
    # YOLO + policy
    python -m src.play \
        --weights runs/detect/finetune/weights/best.pt \
        --policy checkpoints/policy/best.pt --device mps

    # RT-DETR + same policy
    python -m src.play \
        --weights runs/rtdetr/finetune/weights/best.pt \
        --detector rtdetr --policy checkpoints/policy/best.pt --device mps

    # Detection-only smoke test
    python -m src.play --weights runs/detect/finetune/weights/best.pt --detect-only --dry-run

    # Manual window rect if auto-detect fails
    python -m src.play --yolo ... --policy ... --rect 100,200,1280,720

Controls
--------
- F10: emergency stop (releases all held keys and exits)
- Ctrl+C: clean shutdown (releases keys)

Keys are edge-triggered (press/release only on change) to avoid key repeat spam.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.vpt._common import load_config, resolve_device
from src.vpt.detector import infer_backend, load_detector, predict_detections, boxes_to_detections
from src.vpt.pipeline import detector_weights_from_cfg, resolve_pipeline_config
from src.vpt.runner import Runner

# Matches katana_logger / log_to_actions / vpt_config action_space.buttons
KEY_MAP: dict[str, Any] = {
    "jump": "w",
    "left": "a",
    "down": "s",
    "right": "d",
    "interact": "f",
}
BUTTON_NAMES = ["jump", "left", "down", "right", "interact", "slow_mo", "attack"]


@dataclass
class WindowRect:
    left: int
    top: int
    width: int
    height: int

    def as_mss_region(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


class InputController:
    """Edge-triggered keyboard/mouse injection via pynput."""

    def __init__(self, win: WindowRect, enabled: bool = True) -> None:
        from pynput.keyboard import Controller as Keyboard
        from pynput.mouse import Controller as Mouse

        self.win = win
        self.enabled = enabled
        self.kb = Keyboard()
        self.ms = Mouse()
        self.held: dict[str, bool] = {b: False for b in BUTTON_NAMES}

    def release_all(self) -> None:
        from pynput.keyboard import Key
        from pynput.mouse import Button

        for btn, key in KEY_MAP.items():
            if self.held.get(btn):
                self.kb.release(key)
                self.held[btn] = False
        if self.held.get("slow_mo"):
            self.kb.release(Key.shift)
            self.held["slow_mo"] = False
        if self.held.get("attack"):
            self.ms.release(Button.left)
            self.held["attack"] = False

    def apply(self, action: dict) -> None:
        if not self.enabled:
            return
        from pynput.keyboard import Key
        from pynput.mouse import Button

        if "aim_target" in action:
            tx, ty = action["aim_target"]
            x = self.win.left + tx * self.win.width
            y = self.win.top + ty * self.win.height
            self.ms.position = (int(x), int(y))

        for btn, key in KEY_MAP.items():
            want = bool(action.get(btn, False))
            if want == self.held[btn]:
                continue
            if want:
                self.kb.press(key)
            else:
                self.kb.release(key)
            self.held[btn] = want

        want_slow = bool(action.get("slow_mo", False))
        if want_slow != self.held["slow_mo"]:
            if want_slow:
                self.kb.press(Key.shift)
            else:
                self.kb.release(Key.shift)
            self.held["slow_mo"] = want_slow

        want_atk = bool(action.get("attack", False))
        if want_atk != self.held["attack"]:
            if want_atk:
                self.ms.press(Button.left)
            else:
                self.ms.release(Button.left)
            self.held["attack"] = want_atk


def find_window_macos(title_substr: str) -> WindowRect | None:
    """Find an on-screen window whose name or owner contains title_substr."""
    try:
        from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    except ImportError as e:
        raise ImportError(
            "macOS window lookup needs pyobjc. Install with:\n"
            "  pip install pyobjc-framework-Quartz"
        ) from e

    needle = title_substr.lower()
    best: WindowRect | None = None
    best_area = 0
    for w in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID):
        name = (w.get("kCGWindowName") or "").lower()
        owner = (w.get("kCGWindowOwnerName") or "").lower()
        if needle not in name and needle not in owner:
            continue
        b = w.get("kCGWindowBounds") or {}
        width = int(b.get("Width", 0))
        height = int(b.get("Height", 0))
        if width < 100 or height < 100:
            continue
        area = width * height
        if area > best_area:
            best_area = area
            best = WindowRect(left=int(b.get("X", 0)), top=int(b.get("Y", 0)),
                              width=width, height=height)
    return best


def parse_rect(s: str) -> WindowRect:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--rect must be left,top,width,height")
    return WindowRect(*parts)


def grab_frame_bgr(sct, region: dict) -> np.ndarray:
    import mss

    raw = np.array(sct.grab(region))
    return raw[:, :, :3][:, :, ::-1].copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Katana ZERO agent loop.")
    p.add_argument("--pipeline", type=str, default=None, choices=["yolo", "rtdetr"],
                   help="Use a full pipeline config (yolo or rtdetr). Sets config, detector, policy defaults.")
    p.add_argument("--config", type=Path, default=None, help="VPT config (default: from --pipeline or legacy yaml).")
    p.add_argument("--weights", type=Path, default=None, help="Detector weights (.pt).")
    p.add_argument("--yolo", type=Path, default=None, help="Alias for --weights (YOLO pipeline).")
    p.add_argument("--detector", type=str, default="auto", choices=["auto", "yolo", "rtdetr"],
                   help="Detector backend (default: from --pipeline or infer from path).")
    p.add_argument("--policy", type=Path, default=None, help="Policy checkpoint (.pt).")
    p.add_argument("--detect-only", action="store_true",
                   help="Run detector only; skip policy (implies --dry-run unless --inject).")
    p.add_argument("--device", type=str, default=None, help="mps, cpu, 0, etc.")
    p.add_argument("--window-title", type=str, default="Katana",
                   help="Substring to match game window name/owner (macOS).")
    p.add_argument("--rect", type=str, default=None,
                   help="Manual window rect: left,top,width,height (skips auto-detect).")
    p.add_argument("--fps", type=float, default=30.0, help="Target loop rate.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions; do not send keys/mouse to the game.")
    p.add_argument("--inject", action="store_true",
                   help="With --detect-only, still send keys (not recommended).")
    p.add_argument("--kill-key", type=str, default="f10",
                   help="Emergency stop key name (default f10).")
    p.add_argument("--print-every", type=int, default=30,
                   help="Print action dict every N frames (0 = never).")
    return p.parse_args()


def resolve_window(args: argparse.Namespace) -> WindowRect:
    if args.rect:
        rect = parse_rect(args.rect)
        print(f"[play] using manual rect: {rect}")
        return rect
    if sys.platform != "darwin":
        raise RuntimeError("Auto window detect is macOS-only. Pass --rect left,top,width,height.")
    rect = find_window_macos(args.window_title)
    if rect is None:
        raise RuntimeError(
            f"No on-screen window matching '{args.window_title}'. "
            "Launch Katana ZERO or pass --rect left,top,width,height."
        )
    print(f"[play] window '{args.window_title}' -> left={rect.left} top={rect.top} "
          f"w={rect.width} h={rect.height}")
    return rect


def make_kill_switch(stop_flag: list[bool], key_name: str) -> Any:
    from pynput import keyboard

    def on_press(key):
        try:
            k = key.char.lower() if hasattr(key, "char") and key.char else str(key).lower()
        except AttributeError:
            k = str(key).lower()
        target = key_name.lower().replace("key.", "")
        if k == target or k.endswith(f".{target}"):
            stop_flag[0] = True
            return False
        return True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def main() -> None:
    args = parse_args()
    config_path = resolve_pipeline_config(args.pipeline, args.config)
    cfg = load_config(config_path)

    pipeline_det = (cfg.get("pipeline") or {}).get("detector")
    detector_backend = args.detector if args.detector != "auto" else (pipeline_det or "auto")

    weights = args.weights or args.yolo
    if weights is None:
        w = detector_weights_from_cfg(cfg)
        if w:
            weights = Path(w)
    if weights is None:
        raise SystemExit("Pass --pipeline yolo|rtdetr, or --weights, or --yolo.")

    policy = args.policy
    if policy is None and not args.detect_only:
        policy = Path(cfg["bc"]["ckpt"])

    device = args.device or cfg.get("device", "mps")
    if str(device).startswith("cuda") and not __import__("torch").cuda.is_available():
        device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"

    dry_run = args.dry_run or args.detect_only
    if args.detect_only and args.inject:
        dry_run = False
    if not args.detect_only and (policy is None or not Path(policy).exists()):
        raise FileNotFoundError(
            f"Policy checkpoint required ({policy}). Train with "
            f"train_bc --config {config_path} or use --detect-only."
        )

    win = resolve_window(args)
    stop_flag = [False]
    kill_listener = make_kill_switch(stop_flag, args.kill_key)
    inputs = InputController(win, enabled=not dry_run)

    print(f"[play] device={device}  fps={args.fps}  dry_run={dry_run}")
    print(f"[play] press {args.kill_key.upper()} for emergency stop, Ctrl+C to quit")

    import mss

    backend = infer_backend(weights, detector_backend)  # type: ignore[arg-type]
    print(f"[play] pipeline={args.pipeline or 'custom'}  config={config_path}")
    print(f"[play] detector={backend}  weights={weights}  policy={policy}")

    runner: Runner | None = None
    detect_only_model = None
    if args.detect_only:
        detect_only_model = load_detector(weights, backend)  # type: ignore[arg-type]
    else:
        runner = Runner(
            cfg=cfg,
            detector_weights=weights,
            detector_backend=detector_backend,  # type: ignore[arg-type]
            policy_ckpt=policy,
            device=resolve_device(str(device)),
            imgsz=args.imgsz,
            conf=args.conf,
        )
        runner.reset()

    frame_dt = 1.0 / max(args.fps, 1.0)
    frame_idx = 0

    try:
        with mss.mss() as sct:
            region = win.as_mss_region()
            while not stop_flag[0]:
                t0 = time.perf_counter()
                frame = grab_frame_bgr(sct, region)

                if runner is not None:
                    action = runner.step(frame)
                else:
                    assert detect_only_model is not None
                    res = predict_detections(
                        detect_only_model, frame, device=str(device),
                        imgsz=args.imgsz, conf=args.conf, verbose=False,
                    )[0]
                    n = len(boxes_to_detections(res, max_objects=64))
                    action = {"_detections": n, "_detector": backend}

                if not dry_run:
                    inputs.apply(action)

                if args.print_every and frame_idx % args.print_every == 0:
                    public = {k: v for k, v in action.items() if not str(k).startswith("_")}
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    print(f"frame={frame_idx}  {elapsed_ms:.1f}ms  {public}")

                frame_idx += 1
                sleep = frame_dt - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[play] interrupted")
    finally:
        stop_flag[0] = True
        kill_listener.stop()
        inputs.release_all()
        print(f"[play] stopped after {frame_idx} frames. All keys released.")


if __name__ == "__main__":
    main()
