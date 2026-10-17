"""Baseline 1: MFCC -> small MLP command classifier."""
from __future__ import annotations

import torch
import torch.nn as nn


class MlpClassifier(nn.Module):
    """Fully-connected classifier over a flattened feature sequence.

    Input: (batch, n_frames, n_features)
    Output: (batch, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int,
        n_frames: int,
        n_features: int,
        hidden_layers: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_frames * n_features
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> flatten to (B, T*F)
        flat = x.reshape(x.size(0), -1)
        return self.net(flat)

    @staticmethod
    def from_config(model_cfg: dict, num_classes: int, n_frames: int, n_features: int) -> "MlpClassifier":
        return MlpClassifier(
            num_classes=num_classes,
            n_frames=n_frames,
            n_features=int(model_cfg.get("n_features", n_features)),
            hidden_layers=tuple(model_cfg.get("hidden_layers", [256, 128])),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )
