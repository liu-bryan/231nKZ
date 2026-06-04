"""Compare YOLOv8 and RT-DETR on the same validation split.

Runs Ultralytics' built-in val() for each checkpoint on the dataset defined in
`data.yaml`, then prints detection metrics side-by-side. Optionally benchmarks
mean inference latency on the val image folder.

Usage:
    # After training both models on the same data yaml:
    python -m src.compare_models \
        --data data_grouped.yaml \
        --yolo runs/detect/finetune/weights/best.pt \
        --rtdetr runs/rtdetr/finetune/weights/best.pt \
        --device mps

    # Benchmark inference speed on up to 50 val images:
    python -m src.compare_models --data data_grouped.yaml --device mps --benchmark
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from ultralytics import RTDETR, YOLO


@dataclass
class EvalResult:
    name: str
    weights: str
    map50_95: float
    map50: float
    map75: float
    precision: float
    recall: float
    infer_ms: float | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare YOLOv8 vs RT-DETR on the same val set.")
    p.add_argument("--data", type=Path, default=Path("data_grouped.yaml"),
                   help="Dataset YAML (both models evaluated on its val split).")
    p.add_argument("--yolo", type=Path, default=Path("runs/detect/finetune/weights/best.pt"),
                   help="YOLOv8 checkpoint (.pt).")
    p.add_argument("--rtdetr", type=Path, default=Path("runs/rtdetr/finetune/weights/best.pt"),
                   help="RT-DETR checkpoint (.pt).")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--benchmark", action="store_true",
                   help="Also measure mean inference ms/image on val images.")
    p.add_argument("--benchmark-n", type=int, default=50,
                   help="Max val images to use for the latency benchmark.")
    p.add_argument("--skip-yolo", action="store_true")
    p.add_argument("--skip-rtdetr", action="store_true")
    return p.parse_args()


def load_val_dir(data_yaml: Path) -> Path:
    with data_yaml.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg.get("path", "."))
    val_rel = cfg.get("val", "images/val")
    val_dir = root / val_rel
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Val image dir not found: {val_dir} (from {data_yaml})")
    return val_dir


def val_images(val_dir: Path, limit: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in exts)
    return imgs[:limit]


def evaluate(model, name: str, weights: Path, data: Path, args: argparse.Namespace) -> EvalResult:
    if not weights.exists():
        raise FileNotFoundError(f"{name} weights not found: {weights}")
    print(f"\n[{name}] validating {weights} ...")
    metrics = model.val(
        data=str(data),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )
    box = metrics.box
    return EvalResult(
        name=name,
        weights=str(weights),
        map50_95=float(box.map),
        map50=float(box.map50),
        map75=float(getattr(box, "map75", 0.0)),
        precision=float(box.mp),
        recall=float(box.mr),
    )


def benchmark(model, val_dir: Path, args: argparse.Namespace) -> float:
    imgs = val_images(val_dir, args.benchmark_n)
    if not imgs:
        return float("nan")
    # Warmup
    model.predict(source=str(imgs[0]), imgsz=args.imgsz, device=args.device, verbose=False)
    t0 = time.perf_counter()
    for p in imgs:
        model.predict(source=str(p), imgsz=args.imgsz, device=args.device, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms / len(imgs)


def print_row(label: str, yolo_val: str, rtdetr_val: str, width: int = 14) -> None:
    print(f"  {label:<18} {yolo_val:>{width}} {rtdetr_val:>{width}}")


def print_comparison(yolo: EvalResult | None, rtdetr: EvalResult | None) -> None:
    print("\n" + "=" * 52)
    print("  YOLOv8 vs RT-DETR  (same val split, same data yaml)")
    print("=" * 52)
    hdr_yolo = yolo.name if yolo else "—"
    hdr_rt = rtdetr.name if rtdetr else "—"
    print(f"  {'metric':<18} {hdr_yolo:>14} {hdr_rt:>14}")
    print("  " + "-" * 48)

    def fmt(r: EvalResult | None, attr: str, pct: bool = True) -> str:
        if r is None:
            return "—"
        v = getattr(r, attr)
        return f"{v:.4f}" if not pct else f"{v * 100:.2f}%"

    print_row("mAP50-95", fmt(yolo, "map50_95"), fmt(rtdetr, "map50_95"))
    print_row("mAP50", fmt(yolo, "map50"), fmt(rtdetr, "map50"))
    print_row("mAP75", fmt(yolo, "map75"), fmt(rtdetr, "map75"))
    print_row("precision", fmt(yolo, "precision"), fmt(rtdetr, "precision"))
    print_row("recall", fmt(yolo, "recall"), fmt(rtdetr, "recall"))

    if (yolo and yolo.infer_ms is not None) or (rtdetr and rtdetr.infer_ms is not None):
        y_ms = f"{yolo.infer_ms:.1f} ms" if yolo and yolo.infer_ms is not None else "—"
        r_ms = f"{rtdetr.infer_ms:.1f} ms" if rtdetr and rtdetr.infer_ms is not None else "—"
        print_row("infer (mean/img)", y_ms, r_ms, width=14)

    print("=" * 52)
    if yolo and rtdetr:
        delta = (rtdetr.map50_95 - yolo.map50_95) * 100
        winner = "RT-DETR" if delta > 0 else "YOLOv8" if delta < 0 else "tie"
        print(f"  mAP50-95 delta: {delta:+.2f} pp  ({winner} ahead on detection quality)")
        if yolo.infer_ms and rtdetr.infer_ms:
            speed_ratio = rtdetr.infer_ms / yolo.infer_ms
            faster = "YOLOv8" if speed_ratio > 1 else "RT-DETR"
            print(f"  speed ratio:    RT-DETR is {speed_ratio:.2f}x vs YOLO  ({faster} faster)")
    print()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    yolo_res: EvalResult | None = None
    rtdetr_res: EvalResult | None = None
    val_dir = load_val_dir(args.data)

    if not args.skip_yolo:
        yolo_model = YOLO(str(args.yolo))
        yolo_res = evaluate(yolo_model, "YOLOv8", args.yolo, args.data, args)
        if args.benchmark:
            print(f"[YOLOv8] benchmarking on up to {args.benchmark_n} val images ...")
            yolo_res.infer_ms = benchmark(yolo_model, val_dir, args)

    if not args.skip_rtdetr:
        rtdetr_model = RTDETR(str(args.rtdetr))
        rtdetr_res = evaluate(rtdetr_model, "RT-DETR", args.rtdetr, args.data, args)
        if args.benchmark:
            print(f"[RT-DETR] benchmarking on up to {args.benchmark_n} val images ...")
            rtdetr_res.infer_ms = benchmark(rtdetr_model, val_dir, args)

    print_comparison(yolo_res, rtdetr_res)


if __name__ == "__main__":
    main()
