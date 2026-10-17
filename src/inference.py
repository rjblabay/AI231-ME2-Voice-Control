"""Shared inference logic - runs the Voice Command Model on one clip.

This module is framework-agnostic at the *interface* level: the same
preprocessing + feature-extraction code path is used by the Mac demo (PyTorch
or ONNX) and the Raspberry Pi (ONNX Runtime). Confidence-based rejection
(UNKNOWN / SILENCE) lives here so every deployment behaves identically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio.features import FeatureConfig, features_from_audio, load_feature_config
from .audio.preprocessing import load_and_standardize, rms
from .config import LabelMap, load_commands_config, repo_root


@dataclass
class Prediction:
    label: str
    confidence: float
    latency_ms: float
    raw_probs: dict[str, float]
    rejected: bool = False
    reason: str = ""


class TorchInference:
    """Runs a PyTorch checkpoint on a single audio clip."""

    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        import torch
        from .models.factory import build_model

        self.torch = torch
        self.device = self._pick_device(device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.label_map = LabelMap(ckpt["label_map"])
        n_features = ckpt["n_features"]
        if n_features is None:
            n_features = int(ckpt["model_cfg"].get("n_mels", 40))
        self.model = build_model(
            ckpt["model_cfg"], ckpt["num_classes"], ckpt["n_frames"], n_features
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.feature_cfg = load_feature_config()
        self.feature_kind = ckpt.get("feature_kind", "logmel")
        self.n_features = ckpt.get("n_features")
        self.rejection_threshold = self._load_threshold()
        self.silence_gate = self._silence_gate()

    def _pick_device(self, device: str):
        import torch

        if device in (None, "auto"):
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _load_threshold() -> float:
        try:
            _, thr = load_commands_config(repo_root() / "configs" / "commands.yaml")
            return thr
        except Exception:
            return 0.5

    @staticmethod
    def _silence_gate() -> float:
        return _silence_gate()

    def predict(self, audio: np.ndarray) -> Prediction:
        """Classify a standardized (16 kHz mono float32) waveform."""
        t0 = time.perf_counter()
        feats = features_from_audio(
            audio, self.feature_cfg, kind=self.feature_kind, n_features=self.n_features
        )
        x = self.torch.from_numpy(feats).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(x)
            probs = self.torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        best = int(probs.argmax())
        conf = float(probs[best])
        raw = {self.label_map.to_label(i): float(p) for i, p in enumerate(probs)}

        # Silence gate: very low-energy clips are SILENCE, never a command.
        if rms(audio) < self.silence_gate:
            return Prediction("SILENCE", conf, latency_ms, raw, rejected=True, reason="low_rms")

        # Confidence rejection: low-confidence predictions become UNKNOWN.
        if conf < self.rejection_threshold:
            return Prediction("UNKNOWN", conf, latency_ms, raw, rejected=True, reason="low_confidence")

        return Prediction(self.label_map.to_label(best), conf, latency_ms, raw, rejected=False)


class OnnxInference:
    """Runs an exported ONNX model via onnxruntime (Mac or Raspberry Pi).

    Uses the SAME preprocessing + feature extraction as training, so the
    feature tensor is identical. This is the deployment path for the Pi.
    """

    def __init__(self, onnx_path: str | Path, label_map: LabelMap | None = None, provider: str | None = None):
        import onnxruntime as ort

        self.ort = ort
        self.feature_cfg = load_feature_config()
        self.label_map = label_map or LabelMap.from_config(_commands_cfg())
        self.rejection_threshold = _rejection_threshold()
        self.silence_gate = _silence_gate()

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [provider] if provider else _default_providers()
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, audio: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        feats = features_from_audio(audio, self.feature_cfg, kind="logmel")
        # CNN/DS-CNN ONNX graphs use (B, 1, T, F); recurrent models use
        # (B, T, F). Read the exported graph instead of assuming one family.
        rank = len(self.session.get_inputs()[0].shape)
        x = (feats[None] if rank == 3 else feats[None, None]).astype(np.float32)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = _softmax(logits)
        best = int(probs.argmax())
        conf = float(probs[best])
        raw = {self.label_map.to_label(i): float(p) for i, p in enumerate(probs)}

        if rms(audio) < self.silence_gate:
            return Prediction("SILENCE", conf, latency_ms, raw, rejected=True, reason="low_rms")
        if conf < self.rejection_threshold:
            return Prediction("UNKNOWN", conf, latency_ms, raw, rejected=True, reason="low_confidence")
        return Prediction(self.label_map.to_label(best), conf, latency_ms, raw, rejected=False)


def _commands_cfg() -> dict[str, Any]:
    from .config import load_yaml

    return load_yaml(repo_root() / "configs" / "commands.yaml")


def _rejection_threshold() -> float:
    try:
        _, thr = load_commands_config(repo_root() / "configs" / "commands.yaml")
        return thr
    except Exception:
        return 0.5


def _silence_gate() -> float:
    """RMS energy below which a clip is forced to SILENCE (default 0.01)."""
    try:
        cfg = _commands_cfg()
        return float(cfg.get("silence_gate", {}).get("rms", 0.01))
    except Exception:
        return 0.01


def _default_providers() -> list[str]:
    """Pick the best available onnxruntime execution provider."""
    import onnxruntime as ort

    avail = ort.get_available_providers()
    # CPU is the portable deployment target and avoids CoreML's writable-cache
    # requirement in restricted terminals. Callers can explicitly request
    # CoreML when their Mac environment supports it.
    if "CPUExecutionProvider" in avail:
        return ["CPUExecutionProvider"]
    return avail or ["CPUExecutionProvider"]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_from_file(
    path: str | Path,
    inference: TorchInference | OnnxInference,
) -> Prediction:
    """Convenience: load a WAV file, standardize, and classify."""
    fc = inference.feature_cfg
    audio = load_and_standardize(
        path,
        sample_rate=fc.sample_rate,
        max_duration_sec=fc.max_duration_sec,
        normalize_target=fc.normalize_target,
    )
    return inference.predict(audio)


def main(argv: list[str] | None = None) -> int:
    """Saved-audio ONNX/PyTorch sanity check used before Pi transfer."""
    import argparse
    parser = argparse.ArgumentParser(description="Run one saved audio file through a VCM model")
    parser.add_argument("--model", required=True, help=".onnx or .pt model")
    parser.add_argument("--audio", required=True, help="WAV/other supported audio file")
    parser.add_argument("--labels-config", default=None, help="Optional binary/alternate labels YAML")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum confidence; 0 uses the model's normal argmax decision")
    args = parser.parse_args(argv)
    label_map = None
    if args.labels_config:
        label_map, _ = load_commands_config(args.labels_config)
    engine = OnnxInference(args.model, label_map=label_map) if str(args.model).lower().endswith(".onnx") else TorchInference(args.model)
    engine.rejection_threshold = args.threshold
    pred = predict_from_file(args.audio, engine)
    print(f"model={args.model}\nlabel={pred.label}\nconfidence={pred.confidence:.4f}\nlatency_ms={pred.latency_ms:.2f}\nreason={pred.reason or 'accepted'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
