"""Attach a katana_logger.py JSONL log to an extracted object trajectory.

Time alignment
--------------
The user's `katana_logger.py` is armed/disarmed by the same F9 keypress that
starts/stops OBS, so the `session_start` event timestamp is treated as the
wall-clock time of video frame 0. Frame n is then at:

    t_frame_n = session_start.t + n / fps + offset

`offset` exists because the logger's pynput callback and OBS' hotkey listener
do not fire at exactly the same instant; a few tens of ms drift is normal. If
the policy seems to react one beat late/early, sweep `--offset-ms` by +/- 100.

Action vector layout
--------------------
Default 7-button multi-binary vector for Katana ZERO:

    [jump, left, down, right, interact, slow_mo, attack]
      W     A    S     D     F          Shift    LMB

Override with `--buttons` if you change the config's `action_space.buttons`.

Aim
---
Aim is NOT taken from the mouse log (the logger no longer records cursor
position). Instead it is computed from the trajectory itself as the direction
from the player to the cursor -- both are YOLO-detected objects -- and quantized
into `aim_bins` directions. The result is stored as `aim` (T,) and `aim_mask`
(T,) in the output .npz. Class ids and bin count come from the VPT config
(overridable via CLI).

Usage
-----
    python -m src.vpt.log_to_actions \
        --trajectory data/trajectories/unlabeled/run_2026-05-30_22-42-30.npz \
        --log        logs/run_2026-05-30_22-42-30.jsonl \
        --out        data/trajectories/labeled/run_2026-05-30_22-42-30.npz
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.vpt._common import load_config
from src.vpt.aim import compute_aim

KEY_TO_BUTTON: dict[str, str] = {
    "w": "jump",
    "space": "jump",
    "a": "left",
    "s": "down",
    "d": "right",
    "f": "interact",
    "shift": "slow_mo",
    "shift_l": "slow_mo",
    "shift_r": "slow_mo",
}

DEFAULT_BUTTONS = ["jump", "left", "down", "right", "interact", "slow_mo", "attack"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True,
                   help="Object trajectory .npz from extract_objects.py.")
    p.add_argument("--log", type=Path, required=True, help="katana_logger JSONL.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=float, default=None,
                   help="Video FPS. Falls back to the `fps` field embedded in the trajectory.")
    p.add_argument("--offset-ms", type=float, default=0.0,
                   help="Shift the input timeline by N ms relative to the video "
                        "(+: inputs land later in the video).")
    p.add_argument("--buttons", nargs="+", default=DEFAULT_BUTTONS,
                   help="Output button order; must match configs/vpt_config.yaml action_space.buttons.")
    p.add_argument("--config", type=Path, default=Path("configs/vpt_config.yaml"),
                   help="VPT config; supplies aim_bins and player/cursor class ids.")
    p.add_argument("--aim-bins", type=int, default=None, help="Override action_space.aim_bins.")
    p.add_argument("--player-class-id", type=int, default=None, help="Override observation.player_class_id.")
    p.add_argument("--cursor-class-id", type=int, default=None, help="Override observation.cursor_class_id.")
    return p.parse_args()


def resolve_aim_settings(args: argparse.Namespace) -> tuple[int, int, int]:
    """Return (aim_bins, player_class_id, cursor_class_id), CLI > config > defaults."""
    cfg = load_config(args.config) if args.config and Path(args.config).exists() else {}
    a_space = cfg.get("action_space", {})
    obs = cfg.get("observation", {})
    aim_bins = args.aim_bins if args.aim_bins is not None else int(a_space.get("aim_bins", 0))
    player_id = args.player_class_id if args.player_class_id is not None \
        else int(obs.get("player_class_id", 0))
    cursor_id = args.cursor_class_id if args.cursor_class_id is not None \
        else int(obs.get("cursor_class_id", 1))
    return aim_bins, player_id, cursor_id


def parse_iso(s: str) -> float:
    """Parse the logger's ISO-8601 timestamp (with ms) into UNIX seconds."""
    return datetime.fromisoformat(s).timestamp()


def _normalize_key(raw: str) -> str:
    k = raw.strip().lower()
    if k.startswith("key."):
        k = k[4:]
    if len(k) >= 2 and k[0] == "'" and k[-1] == "'":
        k = k[1:-1]
    return k


def event_button(evt: dict) -> str | None:
    et = evt.get("type")
    if et in ("keydown", "keyup"):
        return KEY_TO_BUTTON.get(_normalize_key(evt.get("key", "")))
    if et in ("mousedown", "mouseup"):
        if "left" in evt.get("button", "").lower():
            return "attack"
    return None


def load_events(log_path: Path) -> tuple[float, list[tuple[float, dict]]]:
    session_start: float | None = None
    events: list[tuple[float, dict]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evt = json.loads(line)
            et = evt.get("type")
            if et == "session_start":
                if session_start is None:
                    session_start = parse_iso(evt["t"])
                continue
            if et == "session_stop":
                continue
            events.append((parse_iso(evt["t"]), evt))
    if session_start is None:
        raise ValueError(f"No session_start event in {log_path}")
    events.sort(key=lambda x: x[0])
    return session_start, events


def main() -> None:
    args = parse_args()

    with np.load(args.trajectory, allow_pickle=False) as d:
        types = d["types"]
        feats = d["feats"]
        mask = d["mask"]
        globals_ = d["globals"] if "globals" in d.files else None
        fps_embedded = float(d["fps"]) if "fps" in d.files else None

    fps = args.fps if args.fps is not None else fps_embedded
    if fps is None or fps <= 0:
        raise ValueError(
            "FPS unknown. Either pass --fps or re-run extract_objects.py so it embeds it.")

    T = types.shape[0]
    session_start, events = load_events(args.log)
    offset = args.offset_ms / 1000.0

    btn_idx = {b: i for i, b in enumerate(args.buttons)}
    unknown_seen: set[str] = set()
    actions = np.zeros((T, len(args.buttons)), dtype=np.float32)
    held: dict[str, bool] = {b: False for b in args.buttons}

    ei = 0
    presses = 0
    for f_idx in range(T):
        frame_t = session_start + f_idx / fps + offset
        while ei < len(events) and events[ei][0] <= frame_t:
            evt = events[ei][1]
            btn = event_button(evt)
            if btn is None:
                if evt.get("type") in ("keydown", "keyup"):
                    unknown_seen.add(_normalize_key(evt.get("key", "")))
            elif btn in held:
                pressed = evt["type"].endswith("down")
                if pressed and not held[btn]:
                    presses += 1
                held[btn] = pressed
            ei += 1
        for b, on in held.items():
            actions[f_idx, btn_idx[b]] = 1.0 if on else 0.0

    save = {"types": types, "feats": feats, "mask": mask, "actions": actions}
    if globals_ is not None:
        save["globals"] = globals_
    save["fps"] = np.float32(fps)

    aim_bins, player_id, cursor_id = resolve_aim_settings(args)
    aim_active = 0
    if aim_bins > 0:
        aim, aim_mask = compute_aim(types, feats, mask, player_id, cursor_id, aim_bins)
        save["aim"] = aim
        save["aim_mask"] = aim_mask
        aim_active = int(aim_mask.sum())

    np.savez_compressed(args.out, **save)

    active = int((actions.sum(axis=-1) > 0).sum())
    per_btn = {b: int(actions[:, i].sum()) for i, b in enumerate(args.buttons)}
    print(f"wrote {args.out}  frames={T}  fps={fps:.3f}")
    print(f"  active_frames={active} ({100 * active / max(1, T):.1f}%)  total_presses={presses}")
    print(f"  per-button held-frame counts: {per_btn}")
    if aim_bins > 0:
        print(f"  aim: {aim_active}/{T} frames have player+cursor detected "
              f"({100 * aim_active / max(1, T):.1f}%), {aim_bins}-way bins")
        if aim_active == 0:
            print(f"  WARNING: no frames have both player(cls={player_id}) and "
                  f"cursor(cls={cursor_id}). Check observation.player_class_id / cursor_class_id.")
    if unknown_seen:
        print(f"  note: ignored unmapped keys: {sorted(unknown_seen)}")


if __name__ == "__main__":
    main()
