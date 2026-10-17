"""Small, separate configurable wake-word inference adapter."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .config import LabelMap, load_yaml, repo_root
from .inference import OnnxInference

class WakeWordInference:
    def __init__(self, model: str | Path, labels_config: str | Path, wake_label: str, threshold: float = 0.5):
        labels = LabelMap.from_config(load_yaml(labels_config))
        if wake_label not in labels.classes:
            raise ValueError(f"Wake label {wake_label!r} is not in {labels_config}")
        self.engine = OnnxInference(model, label_map=labels)
        # Wake detection must not use VCM silence/UNKNOWN rejection semantics.
        self.engine.rejection_threshold = 0.0
        self.engine.silence_gate = -1.0
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Wake threshold must be in [0, 1]")
        self.threshold = float(threshold)
        self.wake_label = wake_label
    def predict(self, audio: np.ndarray):
        """Return the two-class wake-word prediction for one audio window."""
        return self.engine.predict(audio)

    def detect(self, audio: np.ndarray) -> bool:
        """Return true only when the wake-label probability clears the gate."""
        return self.is_wake_prediction(self.predict(audio))

    def is_wake_prediction(self, prediction) -> bool:
        """Apply the calibrated gate to a prediction already computed."""
        return prediction.raw_probs.get(self.wake_label, 0.0) >= self.threshold
