"""Live policy runner: detector -> tracker -> VPTPolicy -> action vector.

Supports YOLOv8 or RT-DETR as the object detector; both feed the same policy.

Example:

    runner = Runner.from_config("configs/vpt_config.yaml",
                                detector_weights="runs/detect/finetune/weights/best.pt",
                                policy_ckpt="checkpoints/policy/best.pt")
    # RT-DETR:
    runner = Runner.from_config(...,
                                detector_weights="runs/rtdetr/finetune/weights/best.pt",
                                detector_backend="rtdetr", ...)

CLI:

    python -m src.vpt.runner --config configs/vpt_config.yaml \
        --weights runs/detect/finetune/weights/best.pt \
        --policy checkpoints/policy/best.pt --source recording.mp4

    python -m src.vpt.runner --weights runs/rtdetr/finetune/weights/best.pt \
        --detector rtdetr --policy checkpoints/policy/best.pt --source recording.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.vpt._common import load_config, num_actions_from_cfg, resolve_device
from src.vpt.aim import bin_to_unit_vector
from src.vpt.detector import DetectorBackend, boxes_to_detections, infer_backend, load_detector, predict_detections
from src.vpt.pipeline import detector_weights_from_cfg, resolve_pipeline_config
from src.vpt.model import VPTPolicy
from src.vpt.tracking import CentroidTracker


class Runner:
    def __init__(
        self,
        cfg: dict,
        policy_ckpt: str | Path,
        detector_weights: str | Path | None = None,
        yolo_weights: str | Path | None = None,
        detector_backend: DetectorBackend = "auto",
        device: torch.device | None = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        threshold: float = 0.5,
        aim_radius: float = 0.25,
    ) -> None:
        weights = detector_weights or yolo_weights
        if weights is None:
            raise ValueError("Pass detector_weights= (or legacy yolo_weights=).")

        self.cfg = cfg
        self.device = device or resolve_device(cfg.get("device", "cuda"))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.threshold = threshold
        self.aim_radius = aim_radius
        self.detector_backend = infer_backend(weights, detector_backend)

        self.action_type = cfg["action_space"]["type"]
        self.buttons: list[str] = list(cfg["action_space"]["buttons"])
        num_actions = num_actions_from_cfg(cfg)

        obs = cfg["observation"]
        mcfg = cfg["model"]
        self.max_objects = obs["max_objects"]
        self.num_object_types = obs["num_object_types"]
        self.aim_bins = int(cfg["action_space"].get("aim_bins", 0))
        self.player_class_id = int(obs.get("player_class_id", 0))

        self.detector = load_detector(weights, self.detector_backend)  # type: ignore[arg-type]
        self.policy = VPTPolicy(
            num_object_types=obs["num_object_types"],
            num_actions=num_actions,
            feat_dim=obs["feat_dim"],
            global_dim=obs.get("global_dim", 0),
            d_model=mcfg["d_model"],
            transformer_layers=mcfg["transformer_layers"],
            transformer_heads=mcfg["transformer_heads"],
            lstm_hidden=mcfg["lstm_hidden"],
            dropout=mcfg.get("dropout", 0.1),
            num_aim_bins=self.aim_bins,
        ).to(self.device)
        state = torch.load(policy_ckpt, map_location=self.device)
        self.policy.load_state_dict(state["state_dict"])
        self.policy.eval()

        self.tracker = CentroidTracker()
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    @classmethod
    def from_config(cls, config_path: str | Path, **kwargs: Any) -> "Runner":
        return cls(cfg=load_config(Path(config_path)), **kwargs)

    def reset(self) -> None:
        self.tracker.reset()
        self.hidden = None

    @torch.no_grad()
    def step(self, frame_bgr: np.ndarray, globals_vec: np.ndarray | None = None) -> dict:
        """Run one observation through detector + policy and return action(s).

        frame_bgr: HxWx3 BGR image (as from cv2.VideoCapture / mss).
        globals_vec: optional (global_dim,) float32 -- required if global_dim > 0.
        """
        result = predict_detections(
            self.detector, frame_bgr, device=str(self.device), imgsz=self.imgsz,
            conf=self.conf, iou=self.iou, verbose=False,
        )[0]
        dets = boxes_to_detections(result, self.max_objects)
        tracked = self.tracker.update(dets)

        types_row = np.full((self.max_objects,), self.num_object_types, dtype=np.int64)
        feats_row = np.zeros((self.max_objects, 6), dtype=np.float32)
        mask_row = np.zeros((self.max_objects,), dtype=bool)
        for j, (_, tr) in enumerate(tracked[: self.max_objects]):
            types_row[j] = tr.cls
            feats_row[j] = [tr.cx, tr.cy, tr.w, tr.h, tr.vx, tr.vy]
            mask_row[j] = True

        types_t = torch.from_numpy(types_row).long().view(1, 1, -1).to(self.device)
        feats_t = torch.from_numpy(feats_row).float().view(1, 1, *feats_row.shape).to(self.device)
        kpm = torch.from_numpy(~mask_row).view(1, 1, -1).to(self.device)
        globals_t = None
        if self.cfg["observation"].get("global_dim", 0) > 0:
            if globals_vec is None:
                raise ValueError("globals_vec required because observation.global_dim > 0")
            globals_t = torch.from_numpy(np.asarray(globals_vec, dtype=np.float32)
                                          ).view(1, 1, -1).to(self.device)

        button_logits, aim_logits, self.hidden = self.policy(
            types_t, feats_t, kpm, globals_t, self.hidden)
        button_logits = button_logits.squeeze(0).squeeze(0)  # (num_actions,)

        n_objects = int(mask_row.sum())

        if self.action_type == "multi_binary":
            probs = button_logits.sigmoid().cpu().numpy()
            pressed = probs > self.threshold
            out = {b: bool(p) for b, p in zip(self.buttons, pressed)} | {
                "_probs": probs.tolist(),
                "_detector": self.detector_backend,
                "_num_objects": n_objects,
            }
        else:
            idx = int(button_logits.argmax(-1).item())
            out = {"action_idx": idx, "action_name": self.buttons[idx],
                   "_probs": button_logits.softmax(-1).cpu().tolist(),
                   "_detector": self.detector_backend}

        if aim_logits is not None:
            out.update(self._decode_aim(aim_logits.squeeze(0).squeeze(0), types_row, feats_row, mask_row))
        return out

    def _decode_aim(self, aim_logits: torch.Tensor, types_row: np.ndarray,
                    feats_row: np.ndarray, mask_row: np.ndarray) -> dict:
        """Turn the aim head's logits into a direction and a target point.

        Returns the predicted bin, its unit (dx, dy) in normalized image space
        (y down), and -- if the player is currently detected -- a suggested
        normalized target point `aim_target` = player_center + radius*(dx, dy).
        Map `aim_target` to screen pixels via your window rect to move the mouse.
        """
        bin_idx = int(aim_logits.argmax(-1).item())
        dx, dy = bin_to_unit_vector(bin_idx, self.aim_bins)
        aim: dict = {"aim_bin": bin_idx, "aim_dir": [dx, dy]}

        player_xy = None
        for j in range(len(types_row)):
            if mask_row[j] and int(types_row[j]) == self.player_class_id:
                player_xy = (float(feats_row[j, 0]), float(feats_row[j, 1]))
                break
        if player_xy is not None:
            tx = min(1.0, max(0.0, player_xy[0] + self.aim_radius * dx))
            ty = min(1.0, max(0.0, player_xy[1] + self.aim_radius * dy))
            aim["aim_target"] = [tx, ty]  # normalized; multiply by window (w, h) for pixels
        return aim


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=str, default=None, choices=["yolo", "rtdetr"],
                   help="Full pipeline: config + default detector + policy paths.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--weights", type=str, default=None, help="Detector .pt (YOLO or RT-DETR).")
    p.add_argument("--yolo", type=str, default=None, help="Alias for --weights (YOLO).")
    p.add_argument("--detector", type=str, default="auto", choices=["auto", "yolo", "rtdetr"])
    p.add_argument("--policy", type=str, required=True)
    p.add_argument("--source", type=str, required=True, help="Video file to replay (no live keypresses).")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    config_path = resolve_pipeline_config(args.pipeline, args.config)
    cfg = load_config(config_path)
    weights = args.weights or args.yolo or detector_weights_from_cfg(cfg)
    if not weights:
        p.error("Pass --pipeline, --weights, or --yolo.")
    policy = args.policy or cfg["bc"]["ckpt"]
    det = args.detector if args.detector != "auto" else (cfg.get("pipeline") or {}).get("detector", "auto")

    import cv2  # local import: only the CLI path needs it

    runner = Runner(
        cfg=cfg, detector_weights=weights, detector_backend=det,  # type: ignore[arg-type]
        policy_ckpt=policy, imgsz=args.imgsz,
    )
    runner.reset()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.source}")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        action = runner.step(frame)
        if frame_idx % 30 == 0:
            print(f"frame={frame_idx}  action={ {k: v for k, v in action.items() if not k.startswith('_')} }")
        frame_idx += 1
    cap.release()


if __name__ == "__main__":
    _cli()
