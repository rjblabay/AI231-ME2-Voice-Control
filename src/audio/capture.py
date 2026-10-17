"""Microphone capture helpers (sounddevice based).

Used by the live Mac demo and the Raspberry Pi inference. Provides:
    * device enumeration (for diagnostics)
    * a context manager that opens a mono 16 kHz input stream
    * push-to-talk recording (guaranteed fallback, no VAD needed)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np

from .vad import record_until_silence


def list_input_devices() -> list[dict]:
    """Return a list of available input (microphone) devices."""
    import sounddevice as sd

    devs = sd.query_devices()
    out = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            out.append(
                {
                    "index": i,
                    "name": d.get("name"),
                    "channels": d.get("max_input_channels"),
                    "default_sr": d.get("default_samplerate"),
                }
            )
    return out


def default_input_device() -> int | None:
    """Index of the system default input device, or None."""
    import sounddevice as sd

    try:
        return int(sd.default.device[0])
    except Exception:
        return None


@contextmanager
def open_input_stream(
    sample_rate: int = 16000,
    channels: int = 1,
    device: int | None = None,
    dtype: str = "float32",
) -> Iterator:
    """Open a mono input stream at ``sample_rate`` as a context manager."""
    import sounddevice as sd

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        device=device,
        dtype=dtype,
    ) as stream:
        yield stream


def record_push_to_talk(
    stream,
    sample_rate: int = 16000,
    max_duration_sec: float = 5.0,
) -> np.ndarray:
    """Record for a fixed duration (push-to-talk fallback).

    Simpler and more robust than VAD; always produces a clip.
    """
    frames = []
    start = time.time()
    while time.time() - start < max_duration_sec:
        data, _overflow = stream.read(int(sample_rate * 0.05))
        frames.append(data[:, 0].astype(np.float32))
    return np.concatenate(frames).astype(np.float32)


def record_vad(
    stream,
    vad_cfg,
) -> np.ndarray | None:
    """Record one utterance using the energy VAD. Returns None if no speech."""
    return record_until_silence(stream, vad_cfg)
