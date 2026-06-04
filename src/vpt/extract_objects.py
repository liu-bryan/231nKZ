"""Run a fine-tuned detector over a video and dump a tracked object trajectory.

Supports YOLOv8 and RT-DETR (same YOLO-format labels, same .npz schema).
Output has no `actions` yet — attach with log_to_actions.py or pseudo_label.py.

Usage:
    # YOLO (default)
    python -m src.vpt.extract_objects \
        --weights runs/detect/finetune/weights/best.pt \
        --source video.mp4 --out data/trajectories/yolo/unlabeled/session.npz

    # RT-DETR
    python -m src.vpt.extract_objects \
        --weights runs/rtdetr/finetune/weights/best.pt \
        --detector rtdetr \
        --source video.mp4 --out data/trajectories/rtdetr/unlabeled/session.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.vpt._common import load_config
from src.vpt.detector import DetectorBackend, boxes_to_detections, infer_backend, load_detector, predict_detections
from src.vpt.pipeline import detector_weights_from_cfg, resolve_pipeline_config
from src.vpt.tracking import CentroidTracker


def detect_fps(source: str) -> float | None:
    """Best-effort FPS probe for a video file via OpenCV."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract a tracked object-list trajectory from a video.")
    p.add_argument("--pipeline", type=str, default=None, choices=["yolo", "rtdetr"],
                   help="Use pipeline config defaults for weights/detector.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--weights", type=str, default=None, help="YOLO or RT-DETR .pt path.")
    p.add_argument("--detector", type=str, default="auto", choices=["auto", "yolo", "rtdetr"],
                   help="Detector backend (default: infer from --weights path).")
    p.add_argument("--source", type=str, required=True, help="Video file (or image dir).")
    p.add_argument("--out", type=Path, required=True, help="Output .npz path.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-objects-per-frame", type=int, default=64,
                   help="Hard cap per frame; lower if you want smaller files.")
    p.add_argument("--max-track-age", type=int, default=5)
    p.add_argument("--max-track-distance", type=float, default=0.08,
                   help="Max normalized centroid distance for matching.")
    p.add_argument("--fps", type=float, default=None,
                   help="Override video FPS; auto-detected from the source file if omitted.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(resolve_pipeline_config(args.pipeline, args.config))
    weights = args.weights or detector_weights_from_cfg(cfg)
    if not weights:
        raise SystemExit("Pass --pipeline yolo|rtdetr or --weights.")
    det_arg = args.detector if args.detector != "auto" else (cfg.get("pipeline") or {}).get("detector", "auto")
    backend = infer_backend(weights, det_arg)  # type: ignore[arg-type]
    model = load_detector(weights, backend)  # type: ignore[arg-type]
    tracker = CentroidTracker(max_age=args.max_track_age, max_distance=args.max_track_distance)

    per_frame_types: list[list[int]] = []
    per_frame_feats: list[list[list[float]]] = []

    stream = predict_detections(
        model, args.source, device=args.device, imgsz=args.imgsz, conf=args.conf,
        iou=args.iou, stream=True, verbose=False,
    )
    for result in stream:
        dets = boxes_to_detections(result, args.max_objects_per_frame)
        tracked = tracker.update(dets)

        types_row: list[int] = []
        feats_row: list[list[float]] = []
        for _, tr in tracked:
            types_row.append(int(tr.cls))
            feats_row.append([tr.cx, tr.cy, tr.w, tr.h, tr.vx, tr.vy])
        per_frame_types.append(types_row)
        per_frame_feats.append(feats_row)

    T = len(per_frame_types)
    if T == 0:
        raise RuntimeError("No frames processed; check --source.")
    M = max(1, max(len(row) for row in per_frame_types))
    types_arr = np.zeros((T, M), dtype=np.int64)
    feats_arr = np.zeros((T, M, 6), dtype=np.float32)
    mask_arr = np.zeros((T, M), dtype=bool)
    for i, (trow, frow) in enumerate(zip(per_frame_types, per_frame_feats)):
        for j, (cls, feat) in enumerate(zip(trow, frow)):
            types_arr[i, j] = cls
            feats_arr[i, j] = feat
            mask_arr[i, j] = True

    fps = args.fps if args.fps is not None else detect_fps(args.source)
    save_kwargs: dict = {"types": types_arr, "feats": feats_arr, "mask": mask_arr, "detector": backend}
    if fps is not None:
        save_kwargs["fps"] = np.float32(fps)
    np.savez_compressed(args.out, **save_kwargs)
    fps_str = f"{fps:.3f}" if fps is not None else "unknown"
    print(f"wrote {args.out}  detector={backend}  frames={T}  "
          f"max_objects_per_frame={M}  fps={fps_str}")


if __name__ == "__main__":
    main()
