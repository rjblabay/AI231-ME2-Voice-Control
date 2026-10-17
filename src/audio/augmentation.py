"""Training-only audio augmentation.

Applied to the waveform *before* feature extraction. These transforms are
never applied to validation or test data (the caller decides). All knobs are
driven by the ``augmentation`` section of ``configs/audio.yaml``.

Supported:
    * random gain
    * time shift
    * additive background noise (random SNR)
    * optional speed perturbation
    * optional pitch perturbation
    * optional simple reverb (exponential-decay impulse response)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_yaml, repo_root
from .preprocessing import normalize_peak

DEFAULT_AUDIO_CFG_PATH = repo_root() / "configs" / "audio.yaml"


def load_augment_config(path: str | None = None) -> dict[str, Any]:
    p = path or DEFAULT_AUDIO_CFG_PATH
    cfg = load_yaml(p)
    return cfg.get("augmentation", {}) or {}


def _rand(a: float, b: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(a, b))


def random_gain(
    audio: np.ndarray, db_range: float, rng: np.random.Generator
) -> np.ndarray:
    db = _rand(-db_range, db_range, rng)
    factor = 10.0 ** (db / 20.0)
    return audio * factor


def time_shift(audio: np.ndarray, max_ms: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    max_shift = int(sr * max_ms / 1000)
    if max_shift <= 0:
        return audio
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(audio, shift)


def add_noise(
    audio: np.ndarray,
    snr_db_range: tuple[float, float],
    sr: int,
    rng: np.random.Generator,
    noise_audio: np.ndarray | None = None,
) -> np.ndarray:
    """Mix supplied ambient audio (or white noise) at a random SNR."""
    lo, hi = snr_db_range
    snr = _rand(lo, hi, rng)
    sig_power = np.mean(audio ** 2) + 1e-12
    noise_power = sig_power / (10.0 ** (snr / 10.0))
    if noise_audio is None:
        noise = rng.standard_normal(audio.shape[0]).astype(np.float32)
    else:
        noise = np.asarray(noise_audio, dtype=np.float32).reshape(-1)
        if len(noise) < len(audio):
            noise = np.resize(noise, len(audio))
        start = int(rng.integers(0, max(1, len(noise) - len(audio) + 1)))
        noise = noise[start : start + len(audio)]
        if len(noise) < len(audio):
            noise = np.pad(noise, (0, len(audio) - len(noise)))
    noise = noise / np.sqrt(np.mean(noise ** 2) + 1e-12) * np.sqrt(noise_power)
    return audio + noise


def speed_perturb(
    audio: np.ndarray, factor_range: tuple[float, float], sr: int, rng: np.random.Generator
) -> np.ndarray:
    """Change playback speed by resampling, preserving length via pad/trim."""
    from scipy.signal import resample

    factor = _rand(factor_range[0], factor_range[1], rng)
    n_new = int(len(audio) / factor)
    if n_new <= 0:
        return audio
    out = resample(audio, n_new)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    elif len(out) > len(audio):
        out = out[: len(audio)]
    return out.astype(np.float32)


def pitch_perturb(
    audio: np.ndarray, max_semitones: int, sr: int, rng: np.random.Generator
) -> np.ndarray:
    """Shift pitch by resampling then restoring length (keeps duration)."""
    semis = int(rng.integers(-max_semitones, max_semitones + 1))
    if semis == 0:
        return audio
    factor = 2.0 ** (semis / 12.0)
    from scipy.signal import resample

    pitched = resample(audio, int(len(audio) * factor))
    # Restore original length (pitch change, same duration).
    restored = resample(pitched, len(audio))
    return restored.astype(np.float32)


def add_reverb(audio: np.ndarray, rt60_max: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Simple exponential-decay reverb via convolution with an IR."""
    from scipy.signal import fftconvolve

    rt60 = _rand(0.05, rt60_max, rng)
    ir_len = int(rt60 * sr)
    if ir_len < 1:
        return audio
    t = np.arange(ir_len) / sr
    ir = np.exp(-6.91 * t / rt60)  # 6.91 ~ ln(1000) => -60 dB at rt60
    ir = ir / (np.sum(ir) + 1e-12)
    wet = fftconvolve(audio, ir, mode="same")
    mix = 0.7 * audio + 0.3 * wet
    return mix.astype(np.float32)


def augment(
    audio: np.ndarray,
    cfg: dict[str, Any],
    sr: int = 16000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a random subset of configured augmentations to ``audio``.

    If ``cfg['enabled']`` is False or the global probability is missed, the
    input is returned unchanged. Output is peak-normalized to stay in range.
    """
    if not cfg or not cfg.get("enabled", False):
        return audio
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > float(cfg.get("p", 0.5)):
        return audio

    out = audio
    if cfg.get("random_gain_db"):
        out = random_gain(out, float(cfg["random_gain_db"]), rng)
    if cfg.get("time_shift_ms"):
        out = time_shift(out, int(cfg["time_shift_ms"]), sr, rng)

    noise = cfg.get("noise", {})
    if noise.get("enabled"):
        snr_range = tuple(noise.get("snr_db", [10, 40]))
        noise_audio = _sample_noise(noise.get("paths", []), sr, rng)
        out = add_noise(out, snr_range, sr, rng, noise_audio=noise_audio)

    speed = cfg.get("speed_perturb", {})
    if speed.get("enabled"):
        out = speed_perturb(out, tuple(speed.get("factor", [0.9, 1.1])), sr, rng)

    pitch = cfg.get("pitch_perturb", {})
    if pitch.get("enabled"):
        out = pitch_perturb(out, int(pitch.get("semitones", 3)), sr, rng)

    reverb = cfg.get("reverb", {})
    if reverb.get("enabled"):
        out = add_reverb(out, float(reverb.get("rt60_max", 0.4)), sr, rng)

    return normalize_peak(out, 0.98)


def _sample_noise(paths: list[str], sr: int, rng: np.random.Generator) -> np.ndarray | None:
    """Choose a random prepared ambient clip, falling back to white noise.

    File discovery happens lazily so a configuration can name the ambient
    directory before it has been prepared on a new machine.
    """
    files: list[Path] = []
    for root in paths:
        p = Path(root)
        if p.exists():
            files.extend(p.rglob("*.wav"))
    if not files:
        return None
    try:
        from .preprocessing import load_and_standardize
        clip = load_and_standardize(files[int(rng.integers(len(files)))], sr, 3.0, normalize=False)
        return clip
    except Exception:
        return None
