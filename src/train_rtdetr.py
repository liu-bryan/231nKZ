"""Fine-tune an RT-DETR detection model on a custom dataset.

RT-DETR (Real-Time DETR) is Ultralytics' transformer detector. It trains on
the same YOLO-format labels as YOLOv8, so you can compare both on identical data.

Usage:
    python -m src.train_rtdetr --data data_grouped.yaml --device mps
    python -m src.train_rtdetr --config configs/train_rtdetr_config.yaml
    python -m src.train_rtdetr --config configs/train_rtdetr_config.yaml --model rtdetr-x.pt --batch 4

CLI args override anything in the YAML config.
"""

from __future__ import annotations

import os

# RT-DETR uses grid_sample in ways that lack MPS backward support on PyTorch.
# Must be set before torch is imported (via ultralytics below).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import RTDETR


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune RT-DETR on a custom detection dataset.")
    p.add_argument("--config", type=Path, default=Path("configs/train_rtdetr_config.yaml"),
                   help="YAML config with training hyperparameters.")
    p.add_argument("--model", type=str, default=None,
                   help="Override: pretrained checkpoint (e.g. rtdetr-l.pt, rtdetr-x.pt).")
    p.add_argument("--data", type=str, default=None, help="Override: dataset YAML path.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="'mps', '0', 'cpu', etc.")
    p.add_argument("--name", type=str, default=None, help="Run name under project/.")
    p.add_argument("--resume", action="store_true", help="Resume from last.pt in the named run.")
    p.add_argument("--no-val", action="store_true", help="Skip the post-training validation pass.")
    p.add_argument("--export", type=str, default=None,
                   help="Optional export format after training (e.g. onnx, torchscript).")
    return p.parse_args()


def merge_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "name": args.name,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if args.resume:
        cfg["resume"] = True
    return cfg


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    cfg = merge_overrides(load_config(args.config), args)

    model_path = cfg.pop("model", "rtdetr-l.pt")
    data_path = cfg.get("data")
    if not data_path:
        raise ValueError("`data` (path to dataset YAML) must be set in the config or via --data.")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    print(f"[rtdetr] base model:      {model_path}")
    print(f"[rtdetr] dataset yaml:    {data_path}")
    print(f"[rtdetr] epochs:          {cfg.get('epochs')}")
    print(f"[rtdetr] imgsz:           {cfg.get('imgsz')}")
    print(f"[rtdetr] batch:           {cfg.get('batch')}")
    print(f"[rtdetr] device:          {cfg.get('device')}")
    if str(cfg.get("device", "")).startswith("mps"):
        print("[rtdetr] note: MPS uses CPU fallback for unsupported ops "
              "(PYTORCH_ENABLE_MPS_FALLBACK=1). Slower than pure MPS but required for RT-DETR.")

    model = RTDETR(model_path)
    results = model.train(**cfg)

    save_dir = Path(getattr(results, "save_dir", cfg.get("project", "runs/rtdetr")))
    best_pt = save_dir / "weights" / "best.pt"
    print(f"[rtdetr] done. best weights -> {best_pt}")

    if not args.no_val:
        print("[val] running final validation on best.pt ...")
        val_model = RTDETR(str(best_pt)) if best_pt.exists() else model
        metrics = val_model.val(data=data_path, imgsz=cfg.get("imgsz", 640), device=cfg.get("device"))
        try:
            print(f"[val] mAP50-95: {metrics.box.map:.4f} | mAP50: {metrics.box.map50:.4f}")
        except AttributeError:
            pass

    if args.export:
        print(f"[export] exporting best.pt to {args.export} ...")
        export_model = RTDETR(str(best_pt)) if best_pt.exists() else model
        out = export_model.export(format=args.export, imgsz=cfg.get("imgsz", 640))
        print(f"[export] wrote {out}")


if __name__ == "__main__":
    main()
