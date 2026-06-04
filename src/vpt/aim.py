"""Aim direction labels derived from YOLO detections.

Because the in-game crosshair is a labeled YOLO class, aim is fully observable:
it is simply the direction from the player to the cursor, both read from the
object list. No mouse log and no inverse-dynamics model are needed for aim --
this helper computes it deterministically for any trajectory (labeled,
unlabeled, or pseudo-labeled).

Convention: feats store normalized image coordinates with y pointing DOWN
(YOLO xywhn). The aim angle is atan2(cursor_y - player_y, cursor_x - player_x),
mapped to [0, 2*pi), then quantized into `num_bins` equal sectors. Bin 0 is
centered on +x (pointing right); bins increase clockwise in image space.

Use the same convention at inference (see runner.py) so predicted bins map back
to the correct on-screen direction.
"""

from __future__ import annotations

import numpy as np


def _first_of_class(types: np.ndarray, mask: np.ndarray, cls: int) -> tuple[np.ndarray, np.ndarray]:
    """For each frame, find the first valid slot whose class == cls.

    Returns (has, idx): `has[t]` is True if the class is present in frame t,
    `idx[t]` is the slot index of its first (highest-confidence) occurrence.
    """
    sel = mask & (types == cls)        # (T, M)
    has = sel.any(axis=1)              # (T,)
    idx = sel.argmax(axis=1)           # (T,) -> first True, or 0 if none
    return has, idx


def compute_aim(
    types: np.ndarray,        # (T, M) int
    feats: np.ndarray,        # (T, M, F) float, F >= 2 with (x, y, ...)
    mask: np.ndarray,         # (T, M) bool, True = real detection
    player_class_id: int,
    cursor_class_id: int,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame aim bin and a validity mask.

    aim:      (T,) int64    direction bin in [0, num_bins); 0 where undefined.
    aim_mask: (T,) bool     True only when both player and cursor are detected
                            and they are not at the exact same point.
    """
    T = types.shape[0]
    aim = np.zeros((T,), dtype=np.int64)
    aim_mask = np.zeros((T,), dtype=bool)
    if T == 0 or num_bins <= 0:
        return aim, aim_mask

    rows = np.arange(T)
    has_p, idx_p = _first_of_class(types, mask, player_class_id)
    has_c, idx_c = _first_of_class(types, mask, cursor_class_id)

    p_xy = feats[rows, idx_p, :2]      # (T, 2)
    c_xy = feats[rows, idx_c, :2]      # (T, 2)
    d = c_xy - p_xy                    # (T, 2)

    both = has_p & has_c
    nonzero = np.abs(d).sum(axis=1) > 1e-9
    valid = both & nonzero

    ang = np.arctan2(d[:, 1], d[:, 0])         # [-pi, pi], y down
    ang = np.mod(ang, 2.0 * np.pi)             # [0, 2*pi)
    bins = np.floor(ang / (2.0 * np.pi) * num_bins).astype(np.int64) % num_bins

    aim[valid] = bins[valid]
    aim_mask[valid] = True
    return aim, aim_mask


def bin_to_unit_vector(bin_idx: int, num_bins: int) -> tuple[float, float]:
    """Inverse of compute_aim's quantization: bin -> unit (dx, dy) at bin center.

    Used at inference to turn a predicted aim bin back into a screen direction.
    """
    ang = (bin_idx + 0.5) / num_bins * 2.0 * np.pi
    return float(np.cos(ang)), float(np.sin(ang))
