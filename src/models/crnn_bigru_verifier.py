"""Conservative CRNN-BiGRU wake-word verifier with attention-statistics pooling."""
from __future__ import annotations

import torch
import torch.nn as nn


class CrnnBiGruVerifierClassifier(nn.Module):
    """Preserve time resolution, then classify attention-weighted mean *and* spread.

    The standard attention mean can make a single short confuser look like a
    wake word.  Including weighted temporal standard deviation gives the head
    evidence about whether the whole acoustic pattern is consistently Astra.
    """

    def __init__(self, num_classes: int, n_frames: int, n_mels: int = 40,
                 conv_channels: tuple[int, ...] = (24, 48, 96), gru_hidden: int = 96,
                 gru_layers: int = 2, dropout: float = 0.30):
        super().__init__()
        del n_frames
        layers: list[nn.Module] = []
        in_ch, mel = 1, n_mels
        for ch in conv_channels:
            layers += [
                nn.Conv2d(in_ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                nn.MaxPool2d((1, 2)), nn.Dropout2d(dropout * 0.5),
            ]
            in_ch, mel = ch, (mel - 2) // 2 + 1
        self.cnn = nn.Sequential(*layers)
        self.gru = nn.GRU(in_ch * mel, gru_hidden, gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0)
        width = gru_hidden * 2
        self.attention = nn.Sequential(nn.Linear(width, gru_hidden), nn.Tanh(), nn.Linear(gru_hidden, 1))
        self.verifier = nn.Sequential(
            nn.Linear(width * 2, width), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(width, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x).permute(0, 2, 1, 3).flatten(2)
        x, _ = self.gru(x)
        weights = torch.softmax(self.attention(x).squeeze(-1), dim=1).unsqueeze(-1)
        mean = torch.sum(x * weights, dim=1)
        variance = torch.sum((x - mean.unsqueeze(1)).square() * weights, dim=1)
        return self.verifier(torch.cat((mean, torch.sqrt(variance + 1e-6)), dim=1))

    @staticmethod
    def from_config(model_cfg: dict, num_classes: int, n_frames: int, n_mels: int) -> "CrnnBiGruVerifierClassifier":
        return CrnnBiGruVerifierClassifier(
            num_classes, n_frames, int(model_cfg.get("n_mels", n_mels)),
            tuple(model_cfg.get("conv_channels", [24, 48, 96])), int(model_cfg.get("gru_hidden", 96)),
            int(model_cfg.get("gru_layers", 2)), float(model_cfg.get("dropout", 0.30)),
        )
