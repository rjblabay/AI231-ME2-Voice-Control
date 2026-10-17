"""Feature extraction - the ONE shared implementation.

Training, evaluation, the Mac demo and the Raspberry Pi inference all call
into this module so the feature tensor is identical everywhere. There is no
other place that computes log-mel or MFCC features.

Outputs are always a fixed-length sequence of feature vectors:

    shape = (n_frames, n_features)

with ``n_frames`` derived from the configured max duration and hop size, so
every sample has the same shape regardless of its original length.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import load_yaml, repo_root
from .preprocessing import load_and_standardize

DEFAULT_AUDIO_CFG_PATH = repo_root() / "configs" / "audio.yaml"


@dataclass(frozen=True)
class FeatureConfig:
    """Immutable feature-extraction settings (loaded from audio.yaml)."""

    sample_rate: int = 16000
    n_mels: int = 40
    window_ms: int = 25
    hop_ms: int = 10
    n_fft: int = 512
    fmin: int = 0
    fmax: int = 8000
    max_duration_sec: float = 3.0
    normalize_target: float = 0.98

    @property
    def window(self) -> int:
        return int(self.sample_rate * self.window_ms / 1000)

    @property
    def hop(self) -> int:
        return int(self.sample_rate * self.hop_ms / 1000)

    @property
    def n_frames(self) -> int:
        """Fixed number of frames for a max-duration clip."""
        return int(self.max_duration_sec * 1000 / self.hop_ms)

    @property
    def n_features(self) -> int:
        return self.n_mels


def load_feature_config(path: str | None = None) -> FeatureConfig:
    """Build a :class:`FeatureConfig` from ``configs/audio.yaml``."""
    p = path or DEFAULT_AUDIO_CFG_PATH
    cfg = load_yaml(p)
    feat = cfg.get("features", {})
    audio = cfg.get("audio", {})
    return FeatureConfig(
        sample_rate=int(feat.get("sample_rate", audio.get("sample_rate", 16000))),
        n_mels=int(feat.get("n_mels", 40)),
        window_ms=int(feat.get("window_ms", 25)),
        hop_ms=int(feat.get("hop_ms", 10)),
        n_fft=int(feat.get("n_fft", 512)),
        fmin=int(feat.get("fmin", 0)),
        fmax=int(feat.get("fmax", 8000)),
        max_duration_sec=float(audio.get("max_duration_sec", 3.0)),
        normalize_target=float(audio.get("normalize_peak", 0.98)),
    )


def _pad_sequence(features: np.ndarray, n_frames: int) -> np.ndarray:
    """Right-pad (or trim) a (T, F) feature matrix to exactly n_frames rows."""
    T, F = features.shape
    if T >= n_frames:
        return features[:n_frames]
    pad = np.zeros((n_frames - T, F), dtype=features.dtype)
    return np.concatenate([features, pad], axis=0)


def extract_logmel(audio: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Compute a padded log-mel spectrogram for a standardized clip.

    ``audio`` must already be 16 kHz mono float32 (see preprocessing).
    Returns shape ``(cfg.n_frames, cfg.n_mels)``.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        win_length=cfg.window,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    # Normalize to a stable dynamic range for the network.
    logmel = (logmel + 80.0) / 80.0  # map typical [-80, 0] dB to [0, 1]
    logmel = np.clip(logmel, 0.0, 1.0)
    logmel = logmel.T.astype(np.float32)  # (time, mel)
    return _pad_sequence(logmel, cfg.n_frames)


def extract_mfcc(audio: np.ndarray, cfg: FeatureConfig, n_mfcc: int = 40) -> np.ndarray:
    """Compute a padded MFCC sequence for a standardized clip.

    Used by the MLP baseline. Returns shape ``(cfg.n_frames, n_mfcc)``.
    """
    import librosa

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=cfg.sample_rate,
        n_mfcc=n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        win_length=cfg.window,
    )
    # z-score each coefficient column for stability.
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True) + 1e-8
    mfcc = ((mfcc - mean) / std).T.astype(np.float32)  # (time, coeff)
    return _pad_sequence(mfcc, cfg.n_frames)


def features_from_audio(
    audio: np.ndarray,
    cfg: FeatureConfig,
    kind: str = "logmel",
    n_features: int | None = None,
) -> np.ndarray:
    """Dispatch helper: pick log-mel or MFCC based on ``kind``."""
    if kind == "logmel":
        return extract_logmel(audio, cfg)
    if kind == "mfcc":
        return extract_mfcc(audio, cfg, n_features or cfg.n_mels)
    raise ValueError(f"Unknown feature kind: {kind!r} (use 'logmel' or 'mfcc')")


def features_from_file(
    path: str,
    cfg: FeatureConfig,
    kind: str = "logmel",
    n_features: int | None = None,
) -> np.ndarray:
    """Load a file, standardize it, and extract features (convenience)."""
    audio = load_and_standardize(
        path,
        sample_rate=cfg.sample_rate,
        max_duration_sec=cfg.max_duration_sec,
        normalize_target=cfg.normalize_target,
    )
    return features_from_audio(audio, cfg, kind=kind, n_features=n_features)
