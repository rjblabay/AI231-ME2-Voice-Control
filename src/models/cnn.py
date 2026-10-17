"""Baseline 2: Log-Mel -> small conventional CNN command classifier."""
from __future__ import annotations

import torch
import torch.nn as nn


class CnnClassifier(nn.Module):
    """Small 2-D CNN over a (time, mel) log-mel spectrogram.

    Input: (batch, 1, n_frames, n_mels)
    Output: (batch, num_classes) logits

    The classifier head size is computed analytically from the input shape so
    the module is fully concrete at construction time (important for ONNX
    export and parameter counting).
    """

    def __init__(
        self,
        num_classes: int,
        n_frames: int,
        n_mels: int = 40,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = 1
        t, m = n_frames, n_mels
        for ch in conv_channels:
            blocks += [
                nn.Conv2d(in_ch, ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                nn.Dropout(dropout),
            ]
            in_ch = ch
            # Conv with same-padding keeps spatial size; pool halves it.
            t = (t - pool_size) // pool_size + 1
            m = (m - pool_size) // pool_size + 1
        self.features = nn.Sequential(*blocks)
        flat_size = in_ch * t * m
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (B, T, F) -> (B, 1, T, F)
            x = x.unsqueeze(1)
        return self.head(self.features(x))

    @staticmethod
    def from_config(model_cfg: dict, num_classes: int, n_frames: int, n_mels: int) -> "CnnClassifier":
        return CnnClassifier(
            num_classes=num_classes,
            n_frames=n_frames,
            n_mels=int(model_cfg.get("n_mels", n_mels)),
            conv_channels=tuple(model_cfg.get("conv_channels", [32, 64])),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            pool_size=int(model_cfg.get("pool_size", 2)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )
