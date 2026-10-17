"""Old-style continuous Mac wake listener, updated for the new Astra model."""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

from src.audio.capture import default_input_device, list_input_devices
from src.audio.preprocessing import normalize_peak, resample, rms, trim_or_pad
from src.config import repo_root
from src.wake_word import WakeWordInference


WAKE_LABEL = "ASTRA"
BUNDLE = repo_root() / "models" / "macbook" / "ASTRA_CRNN_BIGRU_VERIFIER_20261016_184000"


def _status(chunk: np.ndarray, state: str = "[ACTIVE]", prompt: str = 'Say "Astra"') -> None:
    level = min(12, int(rms(chunk) * 500))
    meter = "#" * level + "-" * (12 - level)
    print(f"\r\033[2K  {state} LISTENING FOR WAKE WORD | {prompt} | mic [{meter}]", end="", flush=True)


def _speech_frames(audio: np.ndarray, sample_rate: int, speech_rms: float) -> int:
    frame = max(1, int(sample_rate * 0.030))
    return sum(rms(audio[i:i + frame]) >= speech_rms for i in range(0, len(audio), frame))


def _model_audio(raw: np.ndarray, capture_rate: int, normalize: bool) -> np.ndarray:
    audio = resample(raw, capture_rate, 16000) if capture_rate != 16000 else raw
    audio = trim_or_pad(audio, 16000 * 3)
    return normalize_peak(audio, 0.98) if normalize else audio


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Astra continuous wake-word listener (Mac)")
    p.add_argument("--wake-onnx", type=Path, default=BUNDLE / "ASTRA_CRNN_BIGRU_VERIFIER_20261016_184000.onnx")
    p.add_argument("--labels", type=Path, default=BUNDLE / "astra_labels.yaml")
    p.add_argument("--wake-threshold", type=float, default=0.50)
    p.add_argument("--wake-confirmations", type=int, default=2)
    p.add_argument("--wake-silence-rms", type=float, default=0.005)
    p.add_argument("--wake-speech-rms", type=float, default=0.0)
    p.add_argument("--wake-min-speech-frames", type=int, default=4)
    p.add_argument("--wake-cooldown-sec", type=float, default=2.0)
    p.add_argument("--normalize-wake", action="store_true", help="peak-normalize live windows")
    p.add_argument("--show-meter", action="store_true", help="show the live microphone meter")
    p.add_argument("--debug-wake", action="store_true")
    p.add_argument("--device", type=int)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args(argv)
    if args.list_devices:
        for d in list_input_devices():
            print(f"[{d['index']}] {d['name']} ({d['channels']} ch, {d['default_sr']:.0f} Hz)")
        return 0
    if (not args.wake_onnx.is_file() or not args.labels.is_file() or args.wake_confirmations < 1
            or args.wake_silence_rms < 0 or args.wake_speech_rms < 0 or args.wake_cooldown_sec < 0):
        p.error("check model paths; confirmations >= 1; RMS/cooldown must be non-negative")

    wake = WakeWordInference(args.wake_onnx, args.labels, WAKE_LABEL, args.wake_threshold)
    device = args.device if args.device is not None else default_input_device()
    # Same old behavior: overlapping 3-second wake window evaluated every 0.5 s.
    window, hop = int(args.sample_rate * 3.0), int(args.sample_rate * 0.5)
    positives, cooldown_until = 0, 0.0
    print("Astra Wake Controller (Mac)")
    print(f"  mic: device {device}; capture: {args.sample_rate} Hz; model: {args.wake_onnx.name}")
    print("  [ACTIVE] Wake listener started. Press Ctrl+C to stop.")

    def show_status(chunk: np.ndarray, state: str = "[ACTIVE]", prompt: str = 'Say "Astra"') -> None:
        if args.show_meter:
            _status(chunk, state, prompt)
    # Capture runs in PortAudio's callback thread.  Inference happens in the
    # main thread, so slow feature extraction can never starve the microphone.
    import sounddevice as sd
    buffer = np.zeros(window, dtype=np.float32)
    write_at, captured = 0, 0
    last_chunk = np.zeros(1, dtype=np.float32)
    callback_overflow = False
    next_eval = time.perf_counter()
    lock = threading.Lock()

    def capture(indata, _frames, _time_info, status):
        nonlocal write_at, captured, last_chunk, callback_overflow
        chunk = indata[:, 0].copy()
        if status:
            callback_overflow = True
        if len(chunk) >= window:
            chunk = chunk[-window:]
        with lock:
            end = write_at + len(chunk)
            if end <= window:
                buffer[write_at:end] = chunk
            else:
                first = window - write_at
                buffer[write_at:] = chunk[:first]
                buffer[:end - window] = chunk[first:]
            write_at = end % window
            captured = min(window, captured + len(chunk))
            last_chunk = chunk

    try:
        with sd.InputStream(samplerate=args.sample_rate, channels=1, dtype="float32", device=device,
                            blocksize=max(1, int(args.sample_rate * 0.05)), latency="high", callback=capture):
            while True:
                time.sleep(0.05)
                with lock:
                    ready = captured >= window
                    chunk = last_chunk.copy()
                    # Once full, write_at points to the oldest sample.
                    ring = np.concatenate((buffer[write_at:], buffer[:write_at])) if ready else np.empty(0, dtype=np.float32)
                    overflow = callback_overflow
                    callback_overflow = False
                if overflow:
                    print("\n  [WARNING] microphone overflow")
                now = time.perf_counter()
                # Keep a fixed 0.5-second evaluation cadence without trying
                # to catch up after a slow inference call.
                if now < next_eval:
                    show_status(chunk); continue
                next_eval = now + 0.5
                if now < cooldown_until:
                    positives = 0; show_status(chunk, "[COOLDOWN]", "waiting after wake"); continue
                if not ready:
                    show_status(chunk); continue
                window_rms = rms(ring)
                if window_rms < args.wake_silence_rms:
                    positives = 0; show_status(chunk, "[SILENCE GATE]", "quiet input ignored"); continue
                frames = _speech_frames(ring, args.sample_rate, args.wake_speech_rms) if args.wake_speech_rms else 0
                if args.wake_speech_rms and frames < args.wake_min_speech_frames:
                    positives = 0; show_status(chunk, "[SPEECH GATE]", "no speech segment yet"); continue
                show_status(chunk)
                pred = wake.predict(_model_audio(ring, args.sample_rate, args.normalize_wake))
                score = pred.raw_probs.get(WAKE_LABEL, 0.0)
                if args.debug_wake:
                    print(f"\n  [DEBUG] rms={window_rms:.5f} speech_frames={frames} wake_score={score:.3f} threshold={wake.threshold:.3f}")
                positives = positives + 1 if wake.is_wake_prediction(pred) else 0
                if positives >= args.wake_confirmations:
                    print(f"\n  [DETECTED] Astra ({score:.1%}).")
                    print(f"  [LATENCY] wake-model inference: {pred.latency_ms:.1f} ms (Mac)")
                    positives = 0
                    cooldown_until = time.perf_counter() + args.wake_cooldown_sec
    except KeyboardInterrupt:
        print("\n  Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
