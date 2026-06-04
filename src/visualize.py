"""Visualization tools for labels, model predictions, and VPT trajectories.

Modes
-----
labels     Draw ground-truth boxes from a YOLO dataset.
predict    Run a detector and draw its predictions.
compare    Side-by-side ground truth vs model predictions.
trajectory Overlay tracked objects (+ optional aim arrow) from a .npz on video.

Examples
--------
    # Ground truth on val set
    python -m src.visualize labels --data data_grouped.yaml --split val --show

    # Model predictions
    python -m src.visualize predict --weights runs/detect/finetune/weights/best.pt \
        --source dataset_grouped/images/val --show

    # GT | predictions side-by-side
    python -m src.visualize compare --weights runs/detect/finetune/weights/best.pt \
        --data data_grouped.yaml --split val --show

    # Trajectory on a recording (player->crosshair aim arrow when both detected)
    python -m src.visualize trajectory \
        --trajectory data/trajectories/labeled/session.npz \
        --video recordings/session.mp4 --show

Interactive keys (when --show):  n/Space = next, p = prev, s = save frame, q/Esc = quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.viz.draw import (
    add_banner,
    draw_aim_arrow,
    draw_boxes,
    list_images,
    load_class_names,
    read_gt_boxes,
    resolve_split_dir,
    stitch_horizontal,
    xywhn_to_pixel,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize labels, detections, and trajectories.")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, default=Path("runs/visualize"),
                        help="Directory to save rendered frames.")
    common.add_argument("--show", action="store_true", help="Open an interactive window.")
    common.add_argument("--limit", type=int, default=0, help="Max frames/images (0 = all).")
    common.add_argument("--scale", type=float, default=1.0, help="Display scale factor.")

    pl = sub.add_parser("labels", parents=[common], help="Visualize dataset ground truth.")
    pl.add_argument("--data", type=Path, default=Path("data_grouped.yaml"))
    pl.add_argument("--split", choices=["train", "val"], default="val")

    pp = sub.add_parser("predict", parents=[common], help="Visualize model predictions.")
    pp.add_argument("--weights", type=Path, required=True)
    pp.add_argument("--source", type=Path, required=True, help="Image, directory, or video.")
    pp.add_argument("--imgsz", type=int, default=640)
    pp.add_argument("--conf", type=float, default=0.25)
    pp.add_argument("--iou", type=float, default=0.7)
    pp.add_argument("--device", type=str, default="mps")
    pp.add_argument("--data", type=Path, default=Path("data_grouped.yaml"),
                    help="For class name lookup.")

    pc = sub.add_parser("compare", parents=[common], help="GT vs predictions side-by-side.")
    pc.add_argument("--weights", type=Path, required=True)
    pc.add_argument("--data", type=Path, default=Path("data_grouped.yaml"))
    pc.add_argument("--split", choices=["train", "val"], default="val")
    pc.add_argument("--imgsz", type=int, default=640)
    pc.add_argument("--conf", type=float, default=0.25)
    pc.add_argument("--iou", type=float, default=0.7)
    pc.add_argument("--device", type=str, default="mps")

    pt = sub.add_parser("trajectory", parents=[common], help="Overlay .npz objects on video.")
    pt.add_argument("--trajectory", type=Path, required=True)
    pt.add_argument("--video", type=Path, required=True)
    pt.add_argument("--data", type=Path, default=Path("data_grouped.yaml"),
                    help="For class name lookup.")
    pt.add_argument("--player-id", type=int, default=10)
    pt.add_argument("--cursor-id", type=int, default=2)
    pt.add_argument("--draw-velocity", action="store_true", help="Draw velocity arrows.")
    pt.add_argument("--no-aim", action="store_true", help="Disable player->crosshair aim arrow.")

    return p.parse_args()


def load_model(weights: Path):
    """Load YOLO or RT-DETR based on run directory name / heuristic."""
    w = str(weights)
    if "rtdetr" in w.lower():
        from ultralytics import RTDETR
        return RTDETR(w)
    from ultralytics import YOLO
    return YOLO(w)


def predict_boxes(model, img: np.ndarray, imgsz: int, conf: float, iou: float, device: str):
    result = model.predict(source=img, imgsz=imgsz, conf=conf, iou=iou,
                           device=device, verbose=False)[0]
    h, w = img.shape[:2]
    boxes = []
    if result.boxes is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        for i in range(len(cls)):
            x1, y1, x2, y2 = xyxy[i]
            boxes.append((int(cls[i]), x1, y1, x2, y2, float(scores[i])))
    return boxes


def maybe_scale(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


class FrameViewer:
    """Interactive slideshow with keyboard navigation."""

    def __init__(self, show: bool, out_dir: Path, scale: float) -> None:
        self.show = show
        self.out_dir = out_dir
        self.scale = scale
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

    def display(self, frames: list[tuple[str, np.ndarray]], save_all: bool = False) -> None:
        if not frames:
            print("No frames to display.")
            return
        idx = 0
        saved = 0
        while True:
            name, frame = frames[idx]
            vis = maybe_scale(frame, self.scale)
            if self.show:
                cv2.imshow("visualize", vis)
                key = cv2.waitKey(0 if idx == 0 else 1) & 0xFF
            else:
                key = ord("n")

            out_path = self.out_dir / name if self.out_dir else None
            if save_all and out_path:
                cv2.imwrite(str(out_path), vis)
                saved += 1

            if not self.show:
                idx += 1
                if idx >= len(frames):
                    break
                continue

            if key in (ord("q"), 27):
                break
            if key in (ord("n"), ord(" "), 83):  # n, space, right arrow
                idx = min(idx + 1, len(frames) - 1)
            elif key in (ord("p"), 81):  # p, left arrow
                idx = max(idx - 1, 0)
            elif key == ord("s") and out_path:
                cv2.imwrite(str(out_path), vis)
                print(f"saved {out_path}")
                saved += 1

        if self.show:
            cv2.destroyAllWindows()
        if save_all and saved:
            print(f"saved {saved} frames -> {self.out_dir}")


def run_labels(args: argparse.Namespace) -> None:
    img_dir, lbl_dir = resolve_split_dir(args.data, args.split)
    names = load_class_names(args.data)
    images = list_images(img_dir)
    if args.limit:
        images = images[: args.limit]

    viewer = FrameViewer(args.show, args.out / "labels" / args.split, args.scale)
    frames: list[tuple[str, np.ndarray]] = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = lbl_dir / f"{img_path.stem}.txt"
        boxes = read_gt_boxes(lbl, w, h)
        vis = draw_boxes(img, boxes, names, prefix="GT: ")
        vis = add_banner(vis, f"ground truth | {img_path.name} | {len(boxes)} boxes")
        frames.append((img_path.name, vis))
    viewer.display(frames, save_all=not args.show)


def run_predict(args: argparse.Namespace) -> None:
    names = load_class_names(args.data)
    model = load_model(args.weights)
    viewer = FrameViewer(args.show, args.out / "predict", args.scale)

    source = args.source
    if source.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        _run_predict_video(args, model, names, viewer, source)
        return

    images = list_images(source)
    if args.limit:
        images = images[: args.limit]
    frames = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = predict_boxes(model, img, args.imgsz, args.conf, args.iou, args.device)
        vis = draw_boxes(img, boxes, names)
        vis = add_banner(vis, f"predict | {img_path.name} | {len(boxes)} dets")
        frames.append((img_path.name, vis))
    viewer.display(frames, save_all=not args.show)


def _run_predict_video(args, model, names, viewer, video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.limit and i >= args.limit:
            break
        boxes = predict_boxes(model, frame, args.imgsz, args.conf, args.iou, args.device)
        vis = draw_boxes(frame, boxes, names)
        vis = add_banner(vis, f"predict | frame {i} | {len(boxes)} dets")
        frames.append((f"{video_path.stem}_f{i:05d}.jpg", vis))
        i += 1
    cap.release()
    viewer.display(frames, save_all=not args.show)


def run_compare(args: argparse.Namespace) -> None:
    img_dir, lbl_dir = resolve_split_dir(args.data, args.split)
    names = load_class_names(args.data)
    model = load_model(args.weights)
    images = list_images(img_dir)
    if args.limit:
        images = images[: args.limit]

    viewer = FrameViewer(args.show, args.out / "compare" / args.split, args.scale)
    frames = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_gt_boxes(lbl_dir / f"{img_path.stem}.txt", w, h)
        pred = predict_boxes(model, img, args.imgsz, args.conf, args.iou, args.device)
        left = draw_boxes(img, gt, names, prefix="GT: ")
        left = add_banner(left, "ground truth")
        right = draw_boxes(img, pred, names, prefix="pred: ")
        right = add_banner(right, f"predictions ({args.weights.name})")
        combined = stitch_horizontal(left, right)
        combined = add_banner(combined, img_path.name)
        frames.append((img_path.name, combined))
    viewer.display(frames, save_all=not args.show)


def _centroid_from_row(feats_row: np.ndarray) -> tuple[float, float]:
    cx, cy = float(feats_row[0]), float(feats_row[1])
    return cx, cy


def run_trajectory(args: argparse.Namespace) -> None:
    names = load_class_names(args.data)
    draw_aim = not args.no_aim

    with np.load(args.trajectory, allow_pickle=False) as d:
        types = d["types"]
        feats = d["feats"]
        mask = d["mask"]
        actions = d["actions"] if "actions" in d.files else None

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    viewer = FrameViewer(args.show, args.out / "trajectory", args.scale)
    frames = []
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok or t >= len(types):
            break
        if args.limit and t >= args.limit:
            break
        h, w = frame.shape[:2]
        boxes = []
        player_c = cursor_c = None
        for j in range(types.shape[1]):
            if not mask[t, j]:
                continue
            cls = int(types[t, j])
            cx, cy, bw, bh = feats[t, j, :4]
            x1, y1, x2, y2 = xywhn_to_pixel(cx, cy, bw, bh, w, h)
            boxes.append((cls, x1, y1, x2, y2))
            if cls == args.player_id:
                player_c = (cx * w, cy * h)
            elif cls == args.cursor_id:
                cursor_c = (cx * w, cy * h)
            if args.draw_velocity and feats.shape[-1] >= 6:
                vx, vy = float(feats[t, j, 4]), float(feats[t, j, 5])
                px, py = cx * w, cy * h
                tip = (int(px + vx * w * 8), int(py + vy * h * 8))
                cv2.arrowedLine(frame, (int(px), int(py)), tip, (200, 200, 200), 1, tipLength=0.3)

        vis = draw_boxes(frame, boxes, names)
        if draw_aim and player_c and cursor_c:
            draw_aim_arrow(vis, player_c, cursor_c)

        banner = f"trajectory frame {t}"
        if actions is not None:
            active = [i for i, v in enumerate(actions[t]) if v > 0.5] if actions.ndim == 2 else []
            if active:
                btn_names = ["jump", "left", "down", "right", "interact", "slow_mo", "attack"]
                pressed = [btn_names[i] for i in active if i < len(btn_names)]
                banner += "  buttons: " + "+".join(pressed)
        vis = add_banner(vis, banner)
        frames.append((f"traj_{t:05d}.jpg", vis))
        t += 1
    cap.release()
    viewer.display(frames, save_all=not args.show)


def main() -> None:
    args = parse_args()
    if args.mode == "labels":
        run_labels(args)
    elif args.mode == "predict":
        run_predict(args)
    elif args.mode == "compare":
        run_compare(args)
    elif args.mode == "trajectory":
        run_trajectory(args)


if __name__ == "__main__":
    main()
