"""Audio loading and standardization.

Every audio path in the project funnels through :func:`load_and_standardize`
so that downstream code always sees the same representation:

    * sample rate  16,000 Hz
    * mono
    * float32 in [-1, 1]
    * peak-normalized (safe)
    * trimmed / zero-padded to ``max_duration_sec``

This is the single place resampling and channel mixing happen. Do not
duplicate this logic elsewhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SAMPLE_RATE = 16000


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample ``audio`` from ``sr`` to ``target_sr``.

    Uses scipy's high-quality polyphase resampler. No-op if rates match.
    """
    if sr == target_sr:
        return audio
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(target_sr, sr)
    up, down = target_sr // g, sr // g
    return resample_poly(audio, up, down)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse a multichannel array to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def normalize_peak(audio: np.ndarray, target: float = 0.98) -> np.ndarray:
    """Scale ``audio`` so its peak magnitude equals ``target``.

    Silent / near-silent clips are returned unchanged to avoid amplifying
    noise (division by a tiny peak).
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-8:
        return audio
    return audio * (target / peak)


def trim_or_pad(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Trim or zero-pad ``audio`` to exactly ``target_samples`` samples."""
    n = audio.shape[0]
    if n >= target_samples:
        return audio[:target_samples]
    pad = target_samples - n
    return np.pad(audio, (0, pad), mode="constant")


def load_and_standardize(
    path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration_sec: float = 3.0,
    normalize: bool = True,
    normalize_target: float = 0.98,
) -> np.ndarray:
    """Load a WAV/FLAC file and standardize it.

    Returns a 1-D float32 array of length
    ``int(sample_rate * max_duration_sec)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    audio = to_mono(audio)
    audio = resample(audio, sr, sample_rate)

    if normalize:
        audio = normalize_peak(audio, normalize_target)

    target = int(round(sample_rate * max_duration_sec))
    audio = trim_or_pad(audio, target)
    return audio.astype(np.float32)


def rms(audio: np.ndarray) -> float:
    """Root-mean-square energy of a clip (0.0 for empty/silent input)."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))
