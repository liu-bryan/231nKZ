"""Compare two independent pipelines on the same video (evaluation only).

Each frame is processed twice — never merged into one object list:
  Path A: YOLO  -> object list A -> shared policy -> actions
  Path B: RT-DETR -> object list B -> shared policy -> actions

By default both paths load the same bc.ckpt from config (checkpoints/shared/policy/).
Pass --shared-policy to override that checkpoint explicitly.

Usage:
    python -m src.vpt.compare_policy --source recording.mp4 --device mps
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.vpt._common import load_config, resolve_device
from src.vpt.pipeline import CONFIG_PATHS, detector_weights_from_cfg
from src.vpt.runner import Runner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare YOLO vs RT-DETR end-to-end pipelines.")
    p.add_argument("--yolo-config", type=Path, default=CONFIG_PATHS["yolo"])
    p.add_argument("--rtdetr-config", type=Path, default=CONFIG_PATHS["rtdetr"])
    p.add_argument("--yolo", type=Path, default=None, help="Override YOLO detector weights.")
    p.add_argument("--rtdetr", type=Path, default=None, help="Override RT-DETR detector weights.")
    p.add_argument("--shared-policy", type=Path, default=None,
                   help="Ablation: use one policy for both pipelines (not recommended).")
    p.add_argument("--source", type=str, required=True, help="Video file.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-frames", type=int, default=0, help="0 = entire video.")
    p.add_argument("--print-every", type=int, default=60,
                   help="Print per-frame disagreement every N frames (0=never).")
    return p.parse_args()


def public_action(action: dict) -> dict:
    return {k: v for k, v in action.items() if not str(k).startswith("_")}


def buttons_match(a: dict, b: dict, buttons: list[str]) -> bool:
    return all(bool(a.get(btn)) == bool(b.get(btn)) for btn in buttons)


def aim_match(a: dict, b: dict) -> bool:
    if "aim_bin" not in a and "aim_bin" not in b:
        return True
    return a.get("aim_bin") == b.get("aim_bin")


def main() -> None:
    args = parse_args()
    cfg_y = load_config(args.yolo_config)
    cfg_r = load_config(args.rtdetr_config)
    device = resolve_device(args.device or cfg_y.get("device", "mps"))
    buttons = list(cfg_y["action_space"]["buttons"])

    yolo_w = args.yolo or Path(detector_weights_from_cfg(cfg_y) or "runs/detect/finetune/weights/best.pt")
    rtdetr_w = args.rtdetr or Path(detector_weights_from_cfg(cfg_r) or "runs/rtdetr/finetune/weights/best.pt")
    policy_y = args.shared_policy or Path(cfg_y["bc"]["ckpt"])
    policy_r = args.shared_policy or Path(cfg_r["bc"]["ckpt"])

    runners = {
        "yolo": Runner(
            cfg=cfg_y, detector_weights=yolo_w, detector_backend="yolo",
            policy_ckpt=policy_y, device=device, imgsz=args.imgsz, conf=args.conf,
        ),
        "rtdetr": Runner(
            cfg=cfg_r, detector_weights=rtdetr_w, detector_backend="rtdetr",
            policy_ckpt=policy_r, device=device, imgsz=args.imgsz, conf=args.conf,
        ),
    }
    for r in runners.values():
        r.reset()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.source}")

    n_frames = 0
    agree_buttons = 0
    agree_aim = 0
    ms_yolo = 0.0
    ms_rtdetr = 0.0
    det_delta_sum = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and n_frames >= args.max_frames:
            break

        t0 = time.perf_counter()
        act_y = runners["yolo"].step(frame)
        ms_yolo += time.perf_counter() - t0

        t0 = time.perf_counter()
        act_r = runners["rtdetr"].step(frame)
        ms_rtdetr += time.perf_counter() - t0

        det_delta_sum += abs(int(act_y.get("_num_objects", 0)) - int(act_r.get("_num_objects", 0)))

        if buttons_match(act_y, act_r, buttons):
            agree_buttons += 1
        if aim_match(act_y, act_r):
            agree_aim += 1

        if args.print_every and n_frames % args.print_every == 0 and not buttons_match(act_y, act_r, buttons):
            print(f"frame={n_frames}  yolo={public_action(act_y)}  rtdetr={public_action(act_r)}")

        n_frames += 1

    cap.release()
    if n_frames == 0:
        raise RuntimeError("No frames read.")

    print(f"\n=== Pipeline comparison ({n_frames} frames) ===")
    print(f"yolo pipeline:   detector={yolo_w}  policy={policy_y}")
    print(f"rtdetr pipeline: detector={rtdetr_w}  policy={policy_r}")
    if args.shared_policy:
        print(f"(ablation: shared policy {args.shared_policy})")
    print(f"button agreement:  {agree_buttons}/{n_frames} ({100 * agree_buttons / n_frames:.1f}%)")
    print(f"aim_bin agreement: {agree_aim}/{n_frames} ({100 * agree_aim / n_frames:.1f}%)")
    print(f"mean |#objects_yolo - #objects_rtdetr|: {det_delta_sum / n_frames:.2f}")
    print(f"mean ms/frame  yolo={1000 * ms_yolo / n_frames:.1f}  rtdetr={1000 * ms_rtdetr / n_frames:.1f}")


if __name__ == "__main__":
    main()
