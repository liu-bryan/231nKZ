"""Shared utilities for VPT training/inference scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(cfg_device)


def num_actions_from_cfg(cfg: dict) -> int:
    action_type = cfg["action_space"]["type"]
    buttons = cfg["action_space"]["buttons"]
    if action_type == "multi_binary":
        return len(buttons)
    if action_type == "discrete":
        return len(buttons)
    raise ValueError(f"Unknown action_space.type: {action_type}")
