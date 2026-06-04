"""Trajectory datasets for VPT-style training.

On-disk format (one .npz per game session):
    types:    (T, M_file) int            class id; pad slots store num_object_types
    feats:    (T, M_file, F) float32     normalized (x, y, w, h, vx, vy)
    mask:     (T, M_file) bool           True where the slot is a real detection
    actions:  optional (T, A) float32 (multi_binary) or (T,) int (discrete)
    globals:  optional (T, G) float32
    aim:      optional (T,) int       direction bin (player -> cursor)
    aim_mask: optional (T,) bool      True where aim is defined

Each session can pad to a different M_file; this loader re-pads/truncates to a
fixed `max_objects` at sample time so all batches have the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TrajectoryItem:
    types: torch.Tensor              # (T, M) long
    feats: torch.Tensor              # (T, M, F) float
    key_padding_mask: torch.Tensor   # (T, M) bool, True = pad
    actions: torch.Tensor | None     # (T, A) float or (T,) long, or None
    globals_: torch.Tensor | None    # (T, G) float, or None
    source: str
    pseudo: bool


class TrajectoryDataset(Dataset):
    """Yields fixed-length subsequences for behavioral-cloning the policy.

    `roots` may include multiple directories (e.g. labeled + pseudo_labeled).
    Set `pseudo_flag` per root to tag samples for loss weighting.
    """

    def __init__(
        self,
        roots: list[tuple[Path, bool]],     # (dir, is_pseudo)
        sequence_length: int,
        stride: int,
        max_objects: int,
        num_object_types: int,
        require_actions: bool = True,
    ) -> None:
        self.seq_len = sequence_length
        self.stride = stride
        self.max_objects = max_objects
        self.num_object_types = num_object_types
        self.require_actions = require_actions

        self.index: list[tuple[Path, int, bool]] = []
        for root, is_pseudo in roots:
            for p in sorted(Path(root).glob("*.npz")):
                with np.load(p, allow_pickle=False) as d:
                    T = d["types"].shape[0]
                    if require_actions and "actions" not in d.files:
                        continue
                if T == 0:
                    continue
                last_start = max(0, T - sequence_length)
                starts = list(range(0, last_start + 1, max(1, stride))) or [0]
                for s in starts:
                    self.index.append((p, s, is_pseudo))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        path, start, is_pseudo = self.index[idx]
        with np.load(path, allow_pickle=False) as d:
            sl = slice(start, start + self.seq_len)
            types = d["types"][sl]
            feats = d["feats"][sl]
            mask = d["mask"][sl]
            actions = d["actions"][sl] if "actions" in d.files else None
            globals_ = d["globals"][sl] if "globals" in d.files else None
            aim = d["aim"][sl] if "aim" in d.files else None
            aim_mask = d["aim_mask"][sl] if "aim_mask" in d.files else None

        types, feats, mask = self._fit_objects(types, feats, mask)
        types, feats, mask, time_extra = self._fit_time(
            types, feats, mask, {"actions": actions, "globals": globals_,
                                 "aim": aim, "aim_mask": aim_mask})
        actions = time_extra["actions"]
        globals_ = time_extra["globals"]
        aim = time_extra["aim"]
        aim_mask = time_extra["aim_mask"]

        out = {
            "types": torch.from_numpy(np.ascontiguousarray(types)).long(),
            "feats": torch.from_numpy(np.ascontiguousarray(feats)).float(),
            "key_padding_mask": torch.from_numpy(np.ascontiguousarray(~mask)),
            "pseudo": torch.tensor(1.0 if is_pseudo else 0.0),
            "source": str(path.name),
        }
        if actions is not None:
            out["actions"] = torch.from_numpy(np.ascontiguousarray(actions)).float() \
                if actions.dtype.kind == "f" else torch.from_numpy(np.ascontiguousarray(actions)).long()
        if globals_ is not None:
            out["globals"] = torch.from_numpy(np.ascontiguousarray(globals_)).float()
        if aim is not None:
            out["aim"] = torch.from_numpy(np.ascontiguousarray(aim)).long()
        if aim_mask is not None:
            out["aim_mask"] = torch.from_numpy(np.ascontiguousarray(aim_mask)).bool()
        return out

    def _fit_objects(self, types, feats, mask):
        T, M = types.shape
        if M >= self.max_objects:
            return types[:, :self.max_objects], feats[:, :self.max_objects], mask[:, :self.max_objects]
        pad = self.max_objects - M
        types = np.concatenate(
            [types, np.full((T, pad), self.num_object_types, dtype=types.dtype)], axis=1)
        feats = np.concatenate(
            [feats, np.zeros((T, pad, feats.shape[-1]), dtype=feats.dtype)], axis=1)
        mask = np.concatenate(
            [mask, np.zeros((T, pad), dtype=mask.dtype)], axis=1)
        return types, feats, mask

    def _fit_time(self, types, feats, mask, extra: dict):
        """Truncate or zero-pad everything along the time axis to seq_len.

        `extra` maps name -> array-or-None; each is sliced/padded the same way as
        types/feats and returned in a dict with the same keys.
        """
        T = types.shape[0]
        if T >= self.seq_len:
            types = types[:self.seq_len]
            feats = feats[:self.seq_len]
            mask = mask[:self.seq_len]
            out = {k: (v[:self.seq_len] if v is not None else None) for k, v in extra.items()}
            return types, feats, mask, out

        pad = self.seq_len - T
        types = np.concatenate(
            [types, np.full((pad, self.max_objects), self.num_object_types, dtype=types.dtype)], axis=0)
        feats = np.concatenate(
            [feats, np.zeros((pad, self.max_objects, feats.shape[-1]), dtype=feats.dtype)], axis=0)
        mask = np.concatenate([mask, np.zeros((pad, self.max_objects), dtype=mask.dtype)], axis=0)
        out = {}
        for k, v in extra.items():
            if v is None:
                out[k] = None
            else:
                tail = np.zeros((pad,) + v.shape[1:], dtype=v.dtype)
                out[k] = np.concatenate([v, tail], axis=0)
        return types, feats, mask, out


class FramePairDataset(Dataset):
    """Yields ((obs_t, obs_{t+gap}), a_t) pairs for IDM training."""

    def __init__(
        self,
        root: Path,
        max_objects: int,
        num_object_types: int,
        frame_gap: int = 1,
    ) -> None:
        self.root = Path(root)
        self.max_objects = max_objects
        self.num_object_types = num_object_types
        self.frame_gap = frame_gap
        self.index: list[tuple[Path, int]] = []
        for p in sorted(self.root.glob("*.npz")):
            with np.load(p, allow_pickle=False) as d:
                if "actions" not in d.files:
                    continue
                T = d["types"].shape[0]
            for t in range(0, max(0, T - frame_gap)):
                self.index.append((p, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        path, t = self.index[idx]
        with np.load(path, allow_pickle=False) as d:
            types_t, feats_t, mask_t = self._frame(d, t)
            types_tp1, feats_tp1, mask_tp1 = self._frame(d, t + self.frame_gap)
            action = d["actions"][t]
        return {
            "types_t": torch.from_numpy(types_t).long(),
            "feats_t": torch.from_numpy(feats_t).float(),
            "mask_t": torch.from_numpy(~mask_t),
            "types_tp1": torch.from_numpy(types_tp1).long(),
            "feats_tp1": torch.from_numpy(feats_tp1).float(),
            "mask_tp1": torch.from_numpy(~mask_tp1),
            "action": torch.from_numpy(action).float() if action.dtype.kind == "f"
                      else torch.tensor(int(action)).long(),
        }

    def _frame(self, d, t):
        types, feats, mask = d["types"][t], d["feats"][t], d["mask"][t]
        M = types.shape[0]
        if M >= self.max_objects:
            return (np.ascontiguousarray(types[:self.max_objects]),
                    np.ascontiguousarray(feats[:self.max_objects]),
                    np.ascontiguousarray(mask[:self.max_objects]))
        pad = self.max_objects - M
        types = np.concatenate([types, np.full(pad, self.num_object_types, dtype=types.dtype)])
        feats = np.concatenate([feats, np.zeros((pad, feats.shape[-1]), dtype=feats.dtype)])
        mask = np.concatenate([mask, np.zeros(pad, dtype=mask.dtype)])
        return np.ascontiguousarray(types), np.ascontiguousarray(feats), np.ascontiguousarray(mask)
