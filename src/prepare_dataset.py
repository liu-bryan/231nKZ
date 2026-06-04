"""Turn a Label Studio "YOLO with Images" export into an Ultralytics dataset.

Label Studio exports a flat layout:
    export/
      images/        all images
      labels/        one .txt per image (YOLO: class cx cy w h, normalized)
      classes.txt    class names, one per line (index = line number, 0-based)

This script produces the train/val layout Ultralytics expects and writes data.yaml:
    <out>/
      images/train  images/val
      labels/train  labels/val
    data.yaml

It also reports the per-class instance counts and how many images contain BOTH
the player and the crosshair (the frames where aim can be supervised).

Usage:
    python -m src.prepare_dataset \
        --export "/path/to/project-1-at-.../" \
        --out dataset --data-yaml data.yaml --val-fraction 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Capture group key from filenames like "<hash>-2026-05-30_11-45-49_swing1.jpg"
# -> the gameplay-session timestamp "2026-05-30_11-45-49".
DEFAULT_GROUP_REGEX = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label Studio YOLO export -> Ultralytics dataset.")
    p.add_argument("--export", type=Path, required=True, help="Label Studio export directory.")
    p.add_argument("--out", type=Path, default=Path("dataset"), help="Output dataset directory.")
    p.add_argument("--data-yaml", type=Path, default=Path("data.yaml"), help="Output data.yaml path.")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-mode", choices=["random", "group"], default="random",
                   help="random: per-image split. group: keep frames from the same "
                        "session (matched by --group-regex) entirely in train OR val.")
    p.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX,
                   help="Regex whose first group is the session key (used by --split-mode group).")
    p.add_argument("--link", action="store_true",
                   help="Symlink images instead of copying (saves disk; keep the export in place).")
    return p.parse_args()


def split_random(pairs: list[tuple[Path, Path]], val_fraction: float, seed: int):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n_val = max(1, round(len(pairs) * val_fraction))
    return pairs[n_val:], pairs[:n_val]


def split_grouped(pairs: list[tuple[Path, Path]], val_fraction: float, seed: int, regex: str):
    """Assign whole sessions to val until ~val_fraction of images is reached."""
    pat = re.compile(regex)
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for img, lbl in pairs:
        m = pat.search(img.stem)
        key = m.group(1) if m else img.stem  # ungroupable files become their own group
        groups[key].append((img, lbl))

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target_val = max(1, round(len(pairs) * val_fraction))
    val_pairs: list[tuple[Path, Path]] = []
    val_keys: list[str] = []
    for k in keys:
        if len(val_pairs) >= target_val:
            break
        val_pairs.extend(groups[k])
        val_keys.append(k)
    val_set = set(val_keys)
    train_pairs = [p for k in keys if k not in val_set for p in groups[k]]
    print(f"  grouped split: {len(groups)} sessions -> "
          f"{len(val_keys)} in val, {len(groups) - len(val_keys)} in train")
    return train_pairs, val_pairs


def read_classes(export: Path) -> list[str]:
    classes_file = export / "classes.txt"
    names = [ln.strip() for ln in classes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not names:
        raise RuntimeError(f"No class names found in {classes_file}")
    return names


def pair_images_labels(export: Path) -> list[tuple[Path, Path]]:
    images_dir, labels_dir = export / "images", export / "labels"
    pairs = []
    missing = 0
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            missing += 1
            continue
        pairs.append((img, lbl))
    if missing:
        print(f"  note: {missing} images had no matching label .txt (skipped)")
    return pairs


def place(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def label_classes(lbl: Path) -> set[int]:
    out = set()
    for line in lbl.read_text().splitlines():
        line = line.strip()
        if line:
            out.add(int(line.split()[0]))
    return out


def main() -> None:
    args = parse_args()
    names = read_classes(args.export)
    pairs = pair_images_labels(args.export)
    if not pairs:
        raise RuntimeError(f"No image/label pairs found under {args.export}")

    if args.split_mode == "group":
        train_pairs, val_pairs = split_grouped(pairs, args.val_fraction, args.seed, args.group_regex)
    else:
        train_pairs, val_pairs = split_random(pairs, args.val_fraction, args.seed)

    instance_counts: Counter = Counter()
    images_with_class: Counter = Counter()
    player_id = names.index("Player") if "Player" in names else None
    cursor_id = names.index("Crosshair") if "Crosshair" in names else None
    both = 0
    incomplete: list[str] = []

    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for img, lbl in split_pairs:
            place(img, args.out / "images" / split / img.name, args.link)
            place(lbl, args.out / "labels" / split / lbl.name, args.link)
            present = label_classes(lbl)
            for c in present:
                images_with_class[c] += 1
            for line in lbl.read_text().splitlines():
                if line.strip():
                    instance_counts[int(line.split()[0])] += 1
            if player_id in present and cursor_id in present:
                both += 1
            elif player_id is not None and cursor_id is not None:
                miss = [n for n, i in (("player", player_id), ("crosshair", cursor_id))
                        if i not in present]
                incomplete.append(f"{img.name} (missing {', '.join(miss)})")

    write_data_yaml(args.data_yaml, args.out, names)

    print(f"\ndataset -> {args.out.resolve()}  [split-mode: {args.split_mode}]")
    print(f"  train: {len(train_pairs)} images   val: {len(val_pairs)} images   classes: {len(names)}")
    print(f"  data.yaml -> {args.data_yaml.resolve()}")
    print("\nper-class (instances / images):")
    for i, name in enumerate(names):
        flag = ""
        if i == player_id:
            flag = "  <- player_class_id"
        elif i == cursor_id:
            flag = "  <- cursor_class_id"
        print(f"  {i:2d} {name:18s} {instance_counts.get(i, 0):5d} / {images_with_class.get(i, 0):4d}{flag}")

    print(f"\naim coverage: {both}/{len(pairs)} images "
          f"({100 * both / len(pairs):.1f}%) contain BOTH player and crosshair")
    if incomplete:
        print(f"  {len(incomplete)} image(s) lack player and/or crosshair:")
        for s in incomplete:
            print(f"    - {s}")
    if player_id is None or cursor_id is None:
        print("  WARNING: 'Player' and/or 'Crosshair' not in classes.txt -- aim cannot be trained.")
    elif both == 0:
        print("  WARNING: no image has both -- check your labeling; aim will have no supervision.")
    print(f"\nset in configs/vpt_config.yaml:  player_class_id: {player_id}   "
          f"cursor_class_id: {cursor_id}   num_object_types: >= {len(names)}")


def write_data_yaml(path: Path, dataset_dir: Path, names: list[str]) -> None:
    lines = [
        "# Auto-generated by src/prepare_dataset.py",
        f"path: {dataset_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(names)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
