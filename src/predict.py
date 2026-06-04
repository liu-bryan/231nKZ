"""Run inference with a fine-tuned YOLOv8 model.

Usage:
    python -m src.predict --weights runs/detect/finetune/weights/best.pt --source path/to/images_or_video
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLOv8 inference.")
    p.add_argument("--weights", type=str, required=True, help="Path to trained .pt file.")
    p.add_argument("--source", type=str, required=True, help="Image, dir, video, or glob.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save", action="store_true", help="Save annotated outputs.")
    p.add_argument("--save-txt", action="store_true", help="Save YOLO-format predictions.")
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="predict")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
