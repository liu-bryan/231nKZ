"""Shared drawing helpers for detection / trajectory visualization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# BGR palette — distinct colors for up to 20 classes.
PALETTE = [
    (255, 56, 56), (56, 255, 56), (56, 128, 255), (255, 200, 56), (200, 56, 255),
    (56, 255, 255), (255, 128, 180), (128, 255, 128), (180, 180, 255), (255, 180, 100),
    (100, 255, 180), (255, 100, 255), (180, 255, 100), (100, 180, 255), (255, 220, 180),
    (180, 100, 100), (100, 100, 255), (200, 200, 100), (100, 200, 200), (200, 100, 200),
]

HIGHLIGHT = {
    "Player": (0, 255, 0),      # green
    "Crosshair": (0, 165, 255), # orange
}


def load_class_names(data_yaml: Path) -> dict[int, str]:
    with data_yaml.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def resolve_split_dir(data_yaml: Path, split: str) -> tuple[Path, Path]:
    with data_yaml.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg["path"])
    rel = cfg.get(split, f"images/{split}")
    img_dir = root / rel
    # labels/val mirrors images/val
    lbl_dir = root / "labels" / Path(rel).name
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    return img_dir, lbl_dir


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)


def read_gt_boxes(label_path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    """Return list of (cls, x1, y1, x2, y2) in pixel coords."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def xywhn_to_pixel(cx: float, cy: float, bw: float, bh: float, w: int, h: int):
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return x1, y1, x2, y2


def class_color(cls_id: int, name: str) -> tuple[int, int, int]:
    if name in HIGHLIGHT:
        return HIGHLIGHT[name]
    return PALETTE[cls_id % len(PALETTE)]


def draw_box(
    img: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    color: tuple[int, int, int],
    label: str = "",
    thickness: int = 2,
) -> None:
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    ty = max(y1, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, label, (x1 + 2, ty), font, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_boxes(
    img: np.ndarray,
    boxes: list[tuple],
    names: dict[int, str],
    prefix: str = "",
    default_conf: float | None = None,
) -> np.ndarray:
    """Draw boxes. Each box is (cls, x1,y1,x2,y2) or (cls, x1,y1,x2,y2, conf)."""
    out = img.copy()
    for b in boxes:
        cls, x1, y1, x2, y2 = b[:5]
        conf = b[5] if len(b) > 5 else default_conf
        name = names.get(int(cls), str(int(cls)))
        color = class_color(int(cls), name)
        thick = 3 if name in HIGHLIGHT else 2
        lbl = f"{prefix}{name}" if conf is None else f"{prefix}{name} {conf:.2f}"
        draw_box(out, x1, y1, x2, y2, color, lbl, thick)
    return out


def draw_aim_arrow(
    img: np.ndarray,
    player_center: tuple[float, float],
    cursor_center: tuple[float, float],
    color: tuple[int, int, int] = (255, 255, 0),
) -> None:
    p = (int(player_center[0]), int(player_center[1]))
    c = (int(cursor_center[0]), int(cursor_center[1]))
    cv2.arrowedLine(img, p, c, color, 2, tipLength=0.15)
    cv2.circle(img, p, 5, (0, 255, 0), -1)
    cv2.circle(img, c, 5, (0, 165, 255), -1)


def add_banner(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def stitch_horizontal(left: np.ndarray, right: np.ndarray, gap: int = 4) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        left = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    if right.shape[0] != h:
        right = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    sep = np.full((h, gap, 3), 80, dtype=np.uint8)
    return np.hstack([left, sep, right])
