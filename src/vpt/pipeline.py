"""Named detector paths: YOLO vs RT-DETR.

Each path has its own trajectory folders and detector weights. IDM and policy
checkpoints are shared (see configs/vpt_config_*.yaml). Never merge object lists
from both detectors in one observation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

PipelineName = Literal["yolo", "rtdetr"]

CONFIG_PATHS: dict[str, Path] = {
    "yolo": Path("configs/vpt_config_yolo.yaml"),
    "rtdetr": Path("configs/vpt_config_rtdetr.yaml"),
}


def resolve_pipeline_config(name: str | None, config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path
    if name is None:
        return Path("configs/vpt_config.yaml")
    key = name.lower()
    if key not in CONFIG_PATHS:
        raise ValueError(f"Unknown pipeline {name!r}; use yolo or rtdetr.")
    return CONFIG_PATHS[key]


def pipeline_from_cfg(cfg: dict[str, Any]) -> str | None:
    block = cfg.get("pipeline") or {}
    return block.get("name") or block.get("detector")


def detector_weights_from_cfg(cfg: dict[str, Any]) -> str | None:
    block = cfg.get("pipeline") or {}
    return block.get("detector_weights")
