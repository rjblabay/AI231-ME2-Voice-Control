"""Model factory - build the right network family from a config dict."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from .cnn import CnnClassifier
from .crnn_bigru import CrnnBiGruClassifier
from .crnn_bigru_verifier import CrnnBiGruVerifierClassifier
from .dscnn import DsCnnClassifier
from .mlp import MlpClassifier


def build_model(model_cfg: dict[str, Any], num_classes: int, n_frames: int, n_features: int) -> nn.Module:
    """Instantiate a model from its config.

    Args:
        model_cfg: the ``model:`` section of a training config.
        num_classes: number of command classes (from the taxonomy).
        n_frames: fixed feature-frame count (from FeatureConfig).
        n_features: feature width (n_mels for log-mel, n_features for MFCC).
    """
    family = model_cfg.get("family", "dscnn")
    if family == "mlp":
        return MlpClassifier.from_config(model_cfg, num_classes, n_frames, n_features)
    if family == "cnn":
        return CnnClassifier.from_config(model_cfg, num_classes, n_frames, n_features)
    if family == "dscnn":
        return DsCnnClassifier.from_config(model_cfg, num_classes, n_frames, n_features)
    if family == "crnn_bigru":
        return CrnnBiGruClassifier.from_config(model_cfg, num_classes, n_frames, n_features)
    if family == "crnn_bigru_verifier":
        return CrnnBiGruVerifierClassifier.from_config(model_cfg, num_classes, n_frames, n_features)
    raise ValueError(f"Unknown model family: {family!r} (use mlp|cnn|dscnn|crnn_bigru|crnn_bigru_verifier)")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
