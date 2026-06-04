"""VPT-style policy and inverse dynamics model for structured object-list inputs.

Policy (per timestep):
    object list -> ObjectEncoder (transformer over the set of objects)
                -> masked-mean pool   -> per-frame embedding
    embedding   -> LSTM (across time) -> ActionHead

IDM (predict action between two adjacent frames):
    objects_t and objects_{t+1} are concatenated with a frame-id embedding (0/1),
    passed through a single transformer, mean-pooled, then mapped to action logits.

Action space is configurable as either:
    - "multi_binary": each button is an independent Bernoulli (BCE loss); supports
      simultaneously held buttons like left+attack+jump.
    - "discrete":     mutually-exclusive actions (softmax + CE loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObjectEncoder(nn.Module):
    """Transformer encoder over a padded set of objects in one frame."""

    def __init__(
        self,
        num_object_types: int,
        feat_dim: int = 6,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        frame_id_vocab: int = 0,
    ) -> None:
        super().__init__()
        self.padding_idx = num_object_types
        self.type_embed = nn.Embedding(num_object_types + 1, d_model, padding_idx=self.padding_idx)
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.frame_id_embed = nn.Embedding(frame_id_vocab, d_model) if frame_id_vocab > 0 else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(
        self,
        types: torch.Tensor,            # (B, M) int
        feats: torch.Tensor,            # (B, M, F) float
        key_padding_mask: torch.Tensor, # (B, M) bool, True = padding
        frame_ids: torch.Tensor | None = None,  # (B, M) int, only used by IDM
    ) -> torch.Tensor:
        x = self.type_embed(types) + self.feat_proj(feats)
        if frame_ids is not None and self.frame_id_embed is not None:
            x = x + self.frame_id_embed(frame_ids)
        # A fully-masked row (a frame with zero detections, common after time
        # padding) makes attention softmax over all -inf -> NaN. Unmask the first
        # slot for those rows so attention is well-defined; their pooled output is
        # still forced to 0 below because `valid` stays all-zero for them.
        all_masked = key_padding_mask.all(dim=1)
        attn_mask = key_padding_mask.clone()
        attn_mask[all_masked, 0] = False
        out = self.encoder(x, src_key_padding_mask=attn_mask)
        valid = (~key_padding_mask).float().unsqueeze(-1)
        pooled = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled  # (B, d_model)


class VPTPolicy(nn.Module):
    """Per-frame object encoder + LSTM + action head."""

    def __init__(
        self,
        num_object_types: int,
        num_actions: int,
        feat_dim: int = 6,
        global_dim: int = 0,
        d_model: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        lstm_hidden: int = 256,
        dropout: float = 0.1,
        num_aim_bins: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = ObjectEncoder(
            num_object_types, feat_dim, d_model, transformer_layers, transformer_heads, dropout
        )
        self.global_dim = global_dim
        self.frame_proj = nn.Linear(d_model + global_dim, lstm_hidden)
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(lstm_hidden, num_actions)
        # Parallel aim head: softmax over `num_aim_bins` directions. Disabled at 0.
        self.num_aim_bins = num_aim_bins
        self.aim_head = nn.Linear(lstm_hidden, num_aim_bins) if num_aim_bins > 0 else None
        self.lstm_hidden = lstm_hidden
        self.num_actions = num_actions

    def forward(
        self,
        types: torch.Tensor,            # (B, T, M)
        feats: torch.Tensor,            # (B, T, M, F)
        key_padding_mask: torch.Tensor, # (B, T, M)
        globals_: torch.Tensor | None = None,  # (B, T, G)
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor]]:
        B, T, M = types.shape
        flat = self.encoder(
            types.reshape(B * T, M),
            feats.reshape(B * T, M, feats.size(-1)),
            key_padding_mask.reshape(B * T, M),
        ).view(B, T, -1)
        if self.global_dim > 0:
            if globals_ is None:
                raise ValueError("globals_ must be provided when global_dim > 0")
            flat = torch.cat([flat, globals_], dim=-1)
        x = self.frame_proj(flat)
        out, hidden = self.lstm(x, hidden)
        button_logits = self.head(out)                       # (B, T, num_actions)
        aim_logits = self.aim_head(out) if self.aim_head is not None else None  # (B, T, bins)
        return button_logits, aim_logits, hidden


class IDM(nn.Module):
    """Inverse dynamics model: predict the action between two consecutive frames."""

    def __init__(
        self,
        num_object_types: int,
        num_actions: int,
        feat_dim: int = 6,
        d_model: int = 128,
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ObjectEncoder(
            num_object_types, feat_dim, d_model, transformer_layers, transformer_heads, dropout,
            frame_id_vocab=2,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_actions),
        )

    def forward(
        self,
        types_t: torch.Tensor, feats_t: torch.Tensor, mask_t: torch.Tensor,
        types_tp1: torch.Tensor, feats_tp1: torch.Tensor, mask_tp1: torch.Tensor,
    ) -> torch.Tensor:
        types = torch.cat([types_t, types_tp1], dim=1)
        feats = torch.cat([feats_t, feats_tp1], dim=1)
        mask = torch.cat([mask_t, mask_tp1], dim=1)
        frame_ids = torch.cat(
            [torch.zeros_like(types_t), torch.ones_like(types_tp1)], dim=1
        )
        pooled = self.encoder(types, feats, mask, frame_ids)
        return self.head(pooled)  # (B, num_actions)


def action_loss(logits: torch.Tensor, targets: torch.Tensor, action_type: str,
                weight: torch.Tensor | None = None) -> torch.Tensor:
    """Unified loss for both multi_binary (BCE) and discrete (CE) action heads.

    logits:  (..., A) raw scores
    targets: (..., A) float for multi_binary, (...) long for discrete
    weight:  optional per-sample weight broadcastable to the loss shape
    """
    if action_type == "multi_binary":
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        loss = loss.mean(dim=-1)
    elif action_type == "discrete":
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1).long()
        loss = nn.functional.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss = loss.view(targets.shape)
    else:
        raise ValueError(f"Unknown action_type: {action_type}")
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def aim_loss(
    aim_logits: torch.Tensor,   # (B, T, K)
    aim_targets: torch.Tensor,  # (B, T) long, direction bins
    aim_mask: torch.Tensor,     # (B, T) bool/float, 1 where aim is defined
    weight: torch.Tensor | None = None,  # (B, T) or broadcastable per-sample weight
) -> torch.Tensor:
    """Masked cross-entropy for the aim head.

    Only frames where `aim_mask` is set (player + cursor both detected)
    contribute. Returns a zero scalar (still differentiable) if nothing is valid.
    """
    B, T, K = aim_logits.shape
    ce = nn.functional.cross_entropy(
        aim_logits.reshape(B * T, K), aim_targets.reshape(B * T).long(), reduction="none"
    ).view(B, T)
    m = aim_mask.float()
    if weight is not None:
        m = m * weight
    denom = m.sum().clamp_min(1.0)
    return (ce * m).sum() / denom
