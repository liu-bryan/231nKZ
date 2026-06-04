"""Attach recorded gameplay inputs to an extracted trajectory.

Inputs:
    trajectory.npz   -- output of extract_objects.py (no `actions` yet)
    actions.csv      -- one row per frame, header includes the buttons you care
                        about. Values 0/1. The "frame" column is optional; if
                        absent, rows align positionally with the trajectory.
                        Example header (multi_binary, Katana ZERO):
                            frame,left,right,up,down,attack,dodge,slow_mo,interact
                        For "discrete" action spaces, include exactly one column
                        named "action" with integer class ids.

Output is a new .npz at --out that copies types/feats/mask and adds `actions`
(and optionally `globals` if --globals is provided).

Usage:
    python -m src.vpt.attach_actions \
        --trajectory data/trajectories/unlabeled/session_001.npz \
        --actions data/inputs/session_001.csv \
        --buttons left right up down attack dodge slow_mo interact \
        --out data/trajectories/labeled/session_001.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge recorded inputs into a trajectory .npz.")
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--actions", type=Path, required=True, help="CSV of per-frame button states.")
    p.add_argument("--buttons", nargs="+", default=None,
                   help="Button column names in order (multi_binary). "
                        "If omitted and a single 'action' column exists, treat as discrete.")
    p.add_argument("--globals", nargs="*", default=None,
                   help="Optional global-feature column names to copy into `globals`.")
    p.add_argument("--frame-col", default="frame",
                   help="Optional CSV column that holds frame indices; if absent, positional alignment.")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return reader.fieldnames or [], rows


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with np.load(args.trajectory, allow_pickle=False) as d:
        types = d["types"]
        feats = d["feats"]
        mask = d["mask"]
    T = types.shape[0]

    fields, rows = load_csv_rows(args.actions)
    if not rows:
        raise ValueError(f"{args.actions} is empty.")

    if args.frame_col and args.frame_col in fields:
        by_frame: dict[int, dict] = {int(r[args.frame_col]): r for r in rows}
        rows = [by_frame.get(i, {}) for i in range(T)]
    else:
        if len(rows) < T:
            rows = rows + [{}] * (T - len(rows))
        rows = rows[:T]

    if args.buttons:
        actions = np.zeros((T, len(args.buttons)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, b in enumerate(args.buttons):
                val = r.get(b, "0")
                actions[i, j] = float(val) if val not in ("", None) else 0.0
    elif "action" in fields:
        actions = np.zeros((T,), dtype=np.int64)
        for i, r in enumerate(rows):
            actions[i] = int(r.get("action", 0) or 0)
    else:
        raise ValueError("Provide --buttons (multi_binary) or include an 'action' column (discrete).")

    save_kwargs = {"types": types, "feats": feats, "mask": mask, "actions": actions}

    if args.globals:
        globals_arr = np.zeros((T, len(args.globals)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, g in enumerate(args.globals):
                v = r.get(g, "0")
                globals_arr[i, j] = float(v) if v not in ("", None) else 0.0
        save_kwargs["globals"] = globals_arr

    np.savez_compressed(args.out, **save_kwargs)
    n_labeled = int((actions.sum(axis=-1) > 0).sum()) if actions.ndim == 2 else int(np.any(actions != 0))
    print(f"wrote {args.out}  frames={T}  active_action_frames={n_labeled}")


if __name__ == "__main__":
    main()
