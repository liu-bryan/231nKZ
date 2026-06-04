"""Train the inverse dynamics model on labeled (obs_t, obs_{t+gap}) -> a_t pairs.

Once trained, run pseudo_label.py to assign actions to unlabeled trajectories.

Usage:
    python -m src.vpt.train_idm --config configs/vpt_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.vpt._common import load_config, num_actions_from_cfg, resolve_device, set_seed
from src.vpt.pipeline import resolve_pipeline_config
from src.vpt.data import FramePairDataset
from src.vpt.model import IDM, action_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=str, default=None, choices=["yolo", "rtdetr"],
                   help="Use configs/vpt_config_{yolo,rtdetr}.yaml.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--labeled-dir", type=Path, default=None, help="Override config data.labeled_dir.")
    p.add_argument("--ckpt", type=Path, default=None, help="Override config idm.ckpt.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_pipeline_config(args.pipeline, args.config))
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "cuda"))

    obs = cfg["observation"]
    icfg = cfg["idm"]
    action_type = cfg["action_space"]["type"]
    num_actions = num_actions_from_cfg(cfg)

    labeled_dir = args.labeled_dir or Path(cfg["data"]["labeled_dir"])
    dataset = FramePairDataset(
        root=labeled_dir,
        max_objects=obs["max_objects"],
        num_object_types=obs["num_object_types"],
        frame_gap=cfg["data"]["idm_frame_gap"],
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No labeled trajectories found in {labeled_dir}.")

    val_n = max(1, int(0.1 * len(dataset)))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(
        dataset, [train_n, val_n], generator=torch.Generator().manual_seed(cfg.get("seed", 42))
    )
    train_loader = DataLoader(
        train_ds, batch_size=icfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 4), drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=icfg["batch_size"], num_workers=cfg.get("num_workers", 4))

    model = IDM(
        num_object_types=obs["num_object_types"],
        num_actions=num_actions,
        feat_dim=obs["feat_dim"],
        d_model=icfg["d_model"],
        transformer_layers=icfg["transformer_layers"],
        transformer_heads=icfg["transformer_heads"],
        dropout=cfg["model"].get("dropout", 0.1),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=icfg["lr"], weight_decay=icfg["weight_decay"])

    ckpt_path = args.ckpt or Path(icfg["ckpt"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, icfg["epochs"] + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(
                batch["types_t"], batch["feats_t"], batch["mask_t"],
                batch["types_tp1"], batch["feats_tp1"], batch["mask_tp1"],
            )
            loss = action_loss(logits, batch["action"], action_type)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), icfg["grad_clip"])
            opt.step()
            running += loss.item() * logits.size(0)
            seen += logits.size(0)
        train_loss = running / max(1, seen)

        model.eval()
        v_running, v_seen, correct = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits = model(
                    batch["types_t"], batch["feats_t"], batch["mask_t"],
                    batch["types_tp1"], batch["feats_tp1"], batch["mask_tp1"],
                )
                loss = action_loss(logits, batch["action"], action_type)
                v_running += loss.item() * logits.size(0)
                v_seen += logits.size(0)
                if action_type == "discrete":
                    correct += int((logits.argmax(-1) == batch["action"]).sum().item())
                else:
                    pred = (logits.sigmoid() > 0.5).float()
                    correct += int(((pred == batch["action"]).float().mean(dim=-1) > 0.999).sum().item())
        val_loss = v_running / max(1, v_seen)
        acc = correct / max(1, v_seen)
        print(f"[idm] epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  acc={acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "config": cfg}, ckpt_path)
            print(f"[idm]   saved -> {ckpt_path}")

    print(f"[idm] done. best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
