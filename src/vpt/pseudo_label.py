"""Run a trained IDM over unlabeled trajectories and write pseudo-labeled copies.

Reads every .npz in --in-dir, calls the IDM on each consecutive frame pair,
and writes a parallel .npz to --out-dir with an `actions` array attached.
The final frame uses a zero/no-op action (it has no successor to predict from).

Usage:
    python -m src.vpt.pseudo_label --config configs/vpt_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.vpt._common import load_config, num_actions_from_cfg, resolve_device
from src.vpt.pipeline import resolve_pipeline_config
from src.vpt.aim import compute_aim
from src.vpt.model import IDM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=str, default=None, choices=["yolo", "rtdetr"])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--in-dir", type=Path, default=None, help="Override data.unlabeled_dir.")
    p.add_argument("--out-dir", type=Path, default=None, help="Override data.pseudo_labeled_dir.")
    p.add_argument("--ckpt", type=Path, default=None, help="Override idm.ckpt.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="multi_binary: button considered pressed when sigmoid(logit) > this.")
    return p.parse_args()


def pad_frames(types: np.ndarray, feats: np.ndarray, mask: np.ndarray,
               max_objects: int, num_object_types: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, M = types.shape
    if M >= max_objects:
        return types[:, :max_objects], feats[:, :max_objects], mask[:, :max_objects]
    pad = max_objects - M
    types = np.concatenate([types, np.full((T, pad), num_object_types, dtype=types.dtype)], axis=1)
    feats = np.concatenate([feats, np.zeros((T, pad, feats.shape[-1]), dtype=feats.dtype)], axis=1)
    mask = np.concatenate([mask, np.zeros((T, pad), dtype=mask.dtype)], axis=1)
    return types, feats, mask


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_pipeline_config(args.pipeline, args.config))
    device = resolve_device(cfg.get("device", "cuda"))

    obs = cfg["observation"]
    icfg = cfg["idm"]
    action_type = cfg["action_space"]["type"]
    num_actions = num_actions_from_cfg(cfg)

    in_dir = args.in_dir or Path(cfg["data"]["unlabeled_dir"])
    out_dir = args.out_dir or Path(cfg["data"]["pseudo_labeled_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt or Path(icfg["ckpt"])

    state = torch.load(ckpt_path, map_location=device)
    model = IDM(
        num_object_types=obs["num_object_types"],
        num_actions=num_actions,
        feat_dim=obs["feat_dim"],
        d_model=icfg["d_model"],
        transformer_layers=icfg["transformer_layers"],
        transformer_heads=icfg["transformer_heads"],
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    files = sorted(in_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files in {in_dir}")

    gap = cfg["data"]["idm_frame_gap"]

    for src in files:
        with np.load(src, allow_pickle=False) as d:
            types = d["types"]
            feats = d["feats"]
            mask = d["mask"]
            globals_ = d["globals"] if "globals" in d.files else None
        types_p, feats_p, mask_p = pad_frames(
            types, feats, mask, obs["max_objects"], obs["num_object_types"]
        )
        T = types_p.shape[0]
        if T <= gap:
            print(f"[pseudo] {src.name}: too short ({T}), skipping.")
            continue

        if action_type == "multi_binary":
            actions = np.zeros((T, num_actions), dtype=np.float32)
        else:
            actions = np.zeros((T,), dtype=np.int64)

        with torch.no_grad():
            for start in range(0, T - gap, args.batch_size):
                end = min(start + args.batch_size, T - gap)
                idxs = np.arange(start, end)
                t_t = torch.from_numpy(types_p[idxs]).long().to(device)
                f_t = torch.from_numpy(feats_p[idxs]).float().to(device)
                m_t = torch.from_numpy(~mask_p[idxs]).to(device)
                t_tp1 = torch.from_numpy(types_p[idxs + gap]).long().to(device)
                f_tp1 = torch.from_numpy(feats_p[idxs + gap]).float().to(device)
                m_tp1 = torch.from_numpy(~mask_p[idxs + gap]).to(device)
                logits = model(t_t, f_t, m_t, t_tp1, f_tp1, m_tp1)
                if action_type == "multi_binary":
                    actions[idxs] = (logits.sigmoid() > args.threshold).float().cpu().numpy()
                else:
                    actions[idxs] = logits.argmax(-1).cpu().numpy()

        kwargs = {"types": types, "feats": feats, "mask": mask, "actions": actions}
        if globals_ is not None:
            kwargs["globals"] = globals_

        # Aim is directly observable (cursor is a YOLO class), so it is computed
        # deterministically from the object list -- the IDM is not involved.
        aim_bins = int(cfg["action_space"].get("aim_bins", 0))
        aim_active = 0
        if aim_bins > 0:
            aim, aim_mask = compute_aim(
                types, feats, mask,
                int(obs.get("player_class_id", 0)), int(obs.get("cursor_class_id", 1)), aim_bins,
            )
            kwargs["aim"] = aim
            kwargs["aim_mask"] = aim_mask
            aim_active = int(aim_mask.sum())

        out_path = out_dir / src.name
        np.savez_compressed(out_path, **kwargs)
        n_active = int((actions.sum(axis=-1) > 0).sum()) if actions.ndim == 2 else int((actions != 0).sum())
        aim_str = f"  aim_frames={aim_active}" if aim_bins > 0 else ""
        print(f"[pseudo] {src.name}: T={T}  active_frames={n_active}{aim_str}  -> {out_path}")


if __name__ == "__main__":
    main()
