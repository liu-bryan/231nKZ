"""Optional offline augmentation with Albumentations.

YOLOv8 already applies a strong online augmentation pipeline during training
(mosaic, mixup, HSV, flips, affine, etc. -- see configs/train_config.yaml).

Use this module only if you want to *expand* a small dataset on disk before
training (e.g. doubling/tripling its size). It reads YOLO-format labels
(`class cx cy w h`, normalized) and writes augmented image/label pairs into a
sibling directory, keeping bboxes consistent.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_transform(imgsz: int) -> A.Compose:
    """Conservative offline augmentation; strong enough to add variety without
    drifting too far from the original distribution."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                shear=(-3, 3),
                fit_output=False,
                p=0.7,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.GaussNoise(var_limit=(10.0, 40.0)),
                ],
                p=0.3,
            ),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, fill_value=0, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def read_yolo_labels(path: Path) -> tuple[list[list[float]], list[int]]:
    bboxes, labels = [], []
    if not path.exists():
        return bboxes, labels
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = parts[:5]
        labels.append(int(cls))
        bboxes.append([float(cx), float(cy), float(w), float(h)])
    return bboxes, labels


def write_yolo_labels(path: Path, bboxes: list[list[float]], labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}" for c, bb in zip(labels, bboxes)]
    path.write_text("\n".join(lines))


def augment_split(
    images_dir: Path,
    labels_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    n_per_image: int,
    imgsz: int,
    seed: int,
) -> int:
    """Generate `n_per_image` augmented copies for each source image."""
    random.seed(seed)
    transform = build_transform(imgsz)

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = labels_dir / f"{img_path.stem}.txt"
        bboxes, labels = read_yolo_labels(label_path)

        for i in range(n_per_image):
            try:
                out = transform(image=image, bboxes=bboxes, class_labels=labels)
            except Exception:
                continue
            aug_img = cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR)
            aug_bboxes = [list(b) for b in out["bboxes"]]
            aug_labels = [int(c) for c in out["class_labels"]]

            stem = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(out_images_dir / f"{stem}{img_path.suffix}"), aug_img)
            write_yolo_labels(out_labels_dir / f"{stem}.txt", aug_bboxes, aug_labels)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Albumentations augmentation for YOLO datasets.")
    parser.add_argument("--images", type=Path, required=True, help="Source images dir (e.g. dataset/images/train).")
    parser.add_argument("--labels", type=Path, required=True, help="Source labels dir (e.g. dataset/labels/train).")
    parser.add_argument("--out-images", type=Path, required=True, help="Output images dir.")
    parser.add_argument("--out-labels", type=Path, required=True, help="Output labels dir.")
    parser.add_argument("--n", type=int, default=2, help="Augmented copies per source image.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n = augment_split(args.images, args.labels, args.out_images, args.out_labels, args.n, args.imgsz, args.seed)
    print(f"Wrote {n} augmented image/label pairs to {args.out_images}")


if __name__ == "__main__":
    main()
