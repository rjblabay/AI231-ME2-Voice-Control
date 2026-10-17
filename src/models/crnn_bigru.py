"""CRNN with bidirectional GRU pooling for fixed log-Mel inputs."""
from __future__ import annotations

import torch
import torch.nn as nn


class CrnnBiGruClassifier(nn.Module):
    """Frequency-compressing CNN followed by a temporal BiGRU.

    Inputs are log-Mel tensors shaped ``(batch, time, mel)`` (or the
    equivalent 4-D ``(batch, 1, time, mel)`` form).  Pooling is frequency-only
    so the recurrent layer retains the full wake-word time sequence.
    """

    def __init__(
        self,
        num_classes: int,
        n_frames: int,
        n_mels: int = 40,
        conv_channels: tuple[int, ...] = (24, 48, 96),
        gru_hidden: int = 96,
        gru_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        del n_frames  # Sequence length is retained and need not be baked into a layer.
        blocks: list[nn.Module] = []
        in_ch, mel = 1, n_mels
        for ch in conv_channels:
            blocks.extend([
                nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout2d(dropout * 0.5),
            ])
            in_ch = ch
            mel = (mel - 2) // 2 + 1
        self.cnn = nn.Sequential(*blocks)
        self.gru = nn.GRU(
            input_size=in_ch * mel,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden), nn.Tanh(), nn.Linear(gru_hidden, 1)
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(gru_hidden * 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x)                       # B, C, T, F
        x = x.permute(0, 2, 1, 3).flatten(2) # B, T, C*F
        x, _ = self.gru(x)                    # B, T, 2H
        weights = torch.softmax(self.attention(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.head(pooled)

    @staticmethod
    def from_config(model_cfg: dict, num_classes: int, n_frames: int, n_mels: int) -> "CrnnBiGruClassifier":
        return CrnnBiGruClassifier(
            num_classes=num_classes,
            n_frames=n_frames,
            n_mels=int(model_cfg.get("n_mels", n_mels)),
            conv_channels=tuple(model_cfg.get("conv_channels", [24, 48, 96])),
            gru_hidden=int(model_cfg.get("gru_hidden", 96)),
            gru_layers=int(model_cfg.get("gru_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.25)),
        )
