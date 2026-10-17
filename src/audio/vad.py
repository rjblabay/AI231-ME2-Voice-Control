"""Lightweight voice activity detection (energy-based).

Intentionally simple and dependency-free so it works identically on the Mac
and the Raspberry Pi. It is used by the live demo to decide when to start /
stop recording an utterance. It is NOT a neural VAD.

Two primitives:
    * :func:`is_speech_frame` - per-frame energy gate
    * :func:`record_until_silence` - block until speech, record, stop on silence
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .preprocessing import rms


@dataclass
class VadConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    threshold: float = 0.012        # RMS above this counts as speech
    start_timeout_sec: float = 8.0  # how long to wait for speech to begin
    end_silence_sec: float = 0.8    # trailing silence that ends the utterance
    max_duration_sec: float = 6.0   # hard cap on a single recording


def frame_size(cfg: VadConfig) -> int:
    return int(cfg.sample_rate * cfg.frame_ms / 1000)


def is_speech_frame(frame: np.ndarray, threshold: float) -> bool:
    return rms(frame) > threshold


def record_until_silence(stream, cfg: VadConfig) -> np.ndarray | None:
    """Record one utterance from an open sounddevice InputStream.

    ``stream`` must be a 16-bit or float32 mono stream at ``cfg.sample_rate``.
    Blocks until speech is heard, then records until trailing silence or the
    max duration. Returns a float32 mono array, or ``None`` if the caller
    interrupted (returns a sentinel empty array is avoided; use None for
    'no speech').
    """
    fs = frame_size(cfg)
    end_frames = int(cfg.end_silence_sec * 1000 / cfg.frame_ms)
    max_frames = int(cfg.max_duration_sec * 1000 / cfg.frame_ms)

    # --- wait for speech to start ---
    start = time.time()
    while True:
        frame, _ = stream.read(fs)
        frame = frame[:, 0].astype(np.float32)
        if is_speech_frame(frame, cfg.threshold):
            buffer = [frame]
            break
        if time.time() - start > cfg.start_timeout_sec:
            return None

    # --- record until trailing silence ---
    silent_run = 0
    start = time.time()
    while True:
        frame, _ = stream.read(fs)
        frame = frame[:, 0].astype(np.float32)
        buffer.append(frame)
        if not is_speech_frame(frame, cfg.threshold):
            silent_run += 1
        else:
            silent_run = 0
        if silent_run >= end_frames:
            break
        if len(buffer) >= max_frames:
            break
        if time.time() - start > cfg.max_duration_sec:
            break

    return np.concatenate(buffer).astype(np.float32)
