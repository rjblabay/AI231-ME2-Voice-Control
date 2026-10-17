"""Main model: Log-Mel -> DS-CNN (depthwise-separable convolutions).

Depthwise-separable convolutions factor a standard convolution into a
per-channel (depthwise) spatial filter followed by a 1x1 (pointwise)
channel-mixing filter. This drastically cuts parameters and FLOPs versus a
regular conv, which is why DS-CNN is the right shape for a tiny, real-time
voice command model that must run on a Raspberry Pi.

Input:  (batch, 1, n_frames, n_mels)
Output: (batch, num_classes) logits
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    """One depthwise-separable conv block with optional pooling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, pool: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=kernel_size // 2, groups=in_ch
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool) if pool > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        return self.dropout(x)


class DsCnnClassifier(nn.Module):
    """DS-CNN command classifier.

    Configurable stack of depthwise-separable blocks. Kept small (typically
    well under 1M parameters) for edge inference.
    """

    def __init__(
        self,
        num_classes: int,
        n_frames: int,
        n_mels: int = 40,
        blocks: list[dict] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if blocks is None:
            blocks = [
                {"ch": 32, "k": 5, "pool": 2},
                {"ch": 64, "k": 5, "pool": 2},
                {"ch": 128, "k": 5, "pool": 2},
                {"ch": 128, "k": 5, "pool": 1},
            ]
        layers: list[nn.Module] = []
        in_ch = 1
        t, m = n_frames, n_mels
        for b in blocks:
            ch = int(b.get("ch", 32))
            k = int(b.get("k", 5))
            pool = int(b.get("pool", 1))
            layers.append(DepthwiseSeparableBlock(in_ch, ch, k, pool, dropout))
            in_ch = ch
            if pool > 1:
                t = (t - pool) // pool + 1
                m = (m - pool) // pool + 1
        self.features = nn.Sequential(*layers)
        flat_size = in_ch * t * m
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (B, T, F) -> (B, 1, T, F)
            x = x.unsqueeze(1)
        return self.head(self.features(x))

    @staticmethod
    def from_config(model_cfg: dict, num_classes: int, n_frames: int, n_mels: int) -> "DsCnnClassifier":
        return DsCnnClassifier(
            num_classes=num_classes,
            n_frames=n_frames,
            n_mels=int(model_cfg.get("n_mels", n_mels)),
            blocks=model_cfg.get("blocks"),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )
