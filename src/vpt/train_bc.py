"""Behavioral-clone the VPT-style policy on labeled + pseudo-labeled trajectories.

Loss is per-frame BCE (multi_binary) or CE (discrete), averaged over the
sequence. Pseudo-labeled samples are down-weighted by bc.pseudo_label_weight.

Usage:
    python -m src.vpt.train_bc --config configs/vpt_config.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.vpt._common import load_config, num_actions_from_cfg, resolve_device, set_seed
from src.vpt.pipeline import resolve_pipeline_config
from src.vpt.data import TrajectoryDataset
from src.vpt.model import VPTPolicy, action_loss, aim_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=str, default=None, choices=["yolo", "rtdetr"],
                   help="Use configs/vpt_config_{yolo,rtdetr}.yaml.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=None, help="Override bc.ckpt.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Load policy weights from this checkpoint before training.")
    return p.parse_args()


def warmup_cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_pipeline_config(args.pipeline, args.config))
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "cuda"))

    obs = cfg["observation"]
    mcfg = cfg["model"]
    bcfg = cfg["bc"]
    action_type = cfg["action_space"]["type"]
    num_actions = num_actions_from_cfg(cfg)
    aim_bins = int(cfg["action_space"].get("aim_bins", 0))
    aim_w = float(bcfg.get("aim_loss_weight", 1.0))

    roots: list[tuple[Path, bool]] = []
    labeled_dir = Path(cfg["data"]["labeled_dir"])
    pseudo_dir = Path(cfg["data"]["pseudo_labeled_dir"])
    if labeled_dir.exists():
        roots.append((labeled_dir, False))
    if pseudo_dir.exists():
        roots.append((pseudo_dir, True))
    if not roots:
        raise RuntimeError(
            f"Neither {labeled_dir} nor {pseudo_dir} exist. Generate trajectories first.")

    full = TrajectoryDataset(
        roots=roots,
        sequence_length=cfg["data"]["sequence_length"],
        stride=cfg["data"]["stride"],
        max_objects=obs["max_objects"],
        num_object_types=obs["num_object_types"],
        require_actions=True,
    )
    if len(full) == 0:
        raise RuntimeError("No labeled subsequences available.")
    val_n = max(1, int(bcfg["val_fraction"] * len(full)))
    train_n = len(full) - val_n
    train_ds, val_ds = random_split(
        full, [train_n, val_n], generator=torch.Generator().manual_seed(cfg.get("seed", 42))
    )
    train_loader = DataLoader(
        train_ds, batch_size=bcfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 4), drop_last=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bcfg["batch_size"], num_workers=cfg.get("num_workers", 4),
        collate_fn=_collate,
    )

    model = VPTPolicy(
        num_object_types=obs["num_object_types"],
        num_actions=num_actions,
        feat_dim=obs["feat_dim"],
        global_dim=obs.get("global_dim", 0),
        d_model=mcfg["d_model"],
        transformer_layers=mcfg["transformer_layers"],
        transformer_heads=mcfg["transformer_heads"],
        lstm_hidden=mcfg["lstm_hidden"],
        dropout=mcfg.get("dropout", 0.1),
        num_aim_bins=aim_bins,
    ).to(device)

    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["state_dict"], strict=False)
        print(f"[bc] resumed weights from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=bcfg["lr"], weight_decay=bcfg["weight_decay"])
    total_steps = bcfg["epochs"] * max(1, len(train_loader))
    pseudo_w = float(bcfg.get("pseudo_label_weight", 0.5))

    ckpt_path = args.ckpt or Path(bcfg["ckpt"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    best_val = float("inf")
    for epoch in range(1, bcfg["epochs"] + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            lr = warmup_cosine_lr(step, bcfg["warmup_steps"], total_steps, bcfg["lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            types = batch["types"].to(device)
            feats = batch["feats"].to(device)
            mask = batch["key_padding_mask"].to(device)
            actions = batch["actions"].to(device)
            globals_ = batch["globals"].to(device) if "globals" in batch else None
            pseudo = batch["pseudo"].to(device)  # (B,)

            button_logits, aim_logits, _ = model(types, feats, mask, globals_)

            # per-sample weight: 1.0 for labeled, pseudo_w for pseudo-labeled, broadcast to (B, T)
            sample_w = torch.where(pseudo > 0.5, torch.full_like(pseudo, pseudo_w), torch.ones_like(pseudo))
            sample_w = sample_w.view(-1, 1).expand(-1, types.size(1))
            loss = action_loss(button_logits, actions, action_type, weight=sample_w)

            if aim_logits is not None and "aim" in batch:
                aim_t = batch["aim"].to(device)
                aim_m = batch["aim_mask"].to(device)
                loss = loss + aim_w * aim_loss(aim_logits, aim_t, aim_m, weight=sample_w)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), bcfg["grad_clip"])
            opt.step()
            running += loss.item() * types.size(0)
            seen += types.size(0)
            step += 1
        train_loss = running / max(1, seen)

        model.eval()
        v_running, v_seen, correct, total = 0.0, 0, 0, 0
        aim_correct, aim_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                types = batch["types"].to(device)
                feats = batch["feats"].to(device)
                mask = batch["key_padding_mask"].to(device)
                actions = batch["actions"].to(device)
                globals_ = batch["globals"].to(device) if "globals" in batch else None
                button_logits, aim_logits, _ = model(types, feats, mask, globals_)
                loss = action_loss(button_logits, actions, action_type)
                v_running += loss.item() * types.size(0)
                v_seen += types.size(0)
                if action_type == "discrete":
                    correct += int((button_logits.argmax(-1) == actions).sum().item())
                    total += int(actions.numel())
                else:
                    pred = (button_logits.sigmoid() > 0.5).float()
                    correct += int((pred == actions).float().sum().item())
                    total += int(actions.numel())
                if aim_logits is not None and "aim" in batch:
                    aim_t = batch["aim"].to(device)
                    aim_m = batch["aim_mask"].to(device).bool()
                    if aim_m.any():
                        ap = aim_logits.argmax(-1)
                        aim_correct += int(((ap == aim_t) & aim_m).sum().item())
                        aim_total += int(aim_m.sum().item())
        val_loss = v_running / max(1, v_seen)
        acc = correct / max(1, total)
        aim_str = f"  aim_acc={aim_correct / max(1, aim_total):.3f}" if aim_total else ""
        print(f"[bc] epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"acc={acc:.3f}{aim_str}  lr={lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "config": cfg}, ckpt_path)
            print(f"[bc]   saved -> {ckpt_path}")

    print(f"[bc] done. best val loss = {best_val:.4f}")


def _collate(items: list[dict]) -> dict:
    out: dict = {}
    keys = items[0].keys()
    for k in keys:
        vals = [it[k] for it in items]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


if __name__ == "__main__":
    main()
