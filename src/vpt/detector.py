"""Shared Ultralytics detector loading and box parsing for YOLO and RT-DETR.

Both backends emit the same normalized xywh boxes and class ids, so downstream
VPT code (tracker, policy, .npz trajectories) is identical. Use `load_detector`
and `predict_detections` everywhere instead of hard-coding YOLO.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

DetectorBackend = Literal["yolo", "rtdetr", "auto"]
Detection = tuple[int, float, float, float, float]  # cls, cx, cy, w, h (normalized)


def infer_backend(weights: str | Path, backend: DetectorBackend = "auto") -> str:
    """Resolve backend from explicit flag or weights path heuristic."""
    if backend != "auto":
        return backend
    w = str(weights).lower()
    if "rtdetr" in w or "rt-detr" in w:
        return "rtdetr"
    return "yolo"


def load_detector(weights: str | Path, backend: DetectorBackend = "auto") -> Any:
    """Load YOLO or RT-DETR from a .pt checkpoint."""
    resolved = infer_backend(weights, backend)
    path = str(weights)
    if resolved == "rtdetr":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        from ultralytics import RTDETR
        return RTDETR(path)
    from ultralytics import YOLO
    return YOLO(path)


def boxes_to_detections(
    result: Any,
    max_objects: int,
) -> list[Detection]:
    """Parse one Ultralytics predict result into sorted (cls, cx, cy, w, h) list."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xywhn = boxes.xywhn.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    order = np_argsort_conf(boxes)[:max_objects]
    return [
        (int(cls[i]), float(xywhn[i, 0]), float(xywhn[i, 1]),
         float(xywhn[i, 2]), float(xywhn[i, 3]))
        for i in order
    ]


def np_argsort_conf(boxes: Any):
    import numpy as np
    return np.argsort(-boxes.conf.cpu().numpy())


def predict_detections(
    model: Any,
    source: Any,
    *,
    device: str | None = None,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    max_objects: int = 64,
    stream: bool = False,
    verbose: bool = False,
) -> Any:
    """Run predict; returns single result or iterator when stream=True."""
    return model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        stream=stream,
        verbose=verbose,
    )


def predict_frame_detections(
    model: Any,
    frame_bgr: Any,
    *,
    device: str | None = None,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    max_objects: int = 32,
) -> list[Detection]:
    """One frame -> detection list."""
    result = predict_detections(
        model, frame_bgr, device=device, imgsz=imgsz, conf=conf, iou=iou,
        max_objects=max_objects, stream=False, verbose=False,
    )[0]
    return boxes_to_detections(result, max_objects)
