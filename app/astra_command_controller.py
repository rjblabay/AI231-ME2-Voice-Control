"""Continuous Astra wake word + command controller for a Mac microphone.

Run:
    .venv/bin/python -m app.astra_command_controller --device 2

The microphone stays open at 48 kHz.  A rolling three-second window detects
"Astra"; after detection the next three-second utterance is resampled to
16 kHz and classified by the command CRNN–BiGRU ONNX model.
"""
from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path

import numpy as np

from src.audio.capture import default_input_device, list_input_devices
from src.audio.preprocessing import normalize_peak, resample, rms, trim_or_pad
from src.config import load_commands_config, repo_root
from src.inference import OnnxInference
from src.wake_word import WakeWordInference


WAKE_LABEL = "ASTRA"
WAKE_BUNDLE = repo_root() / "models" / "macbook" / "ASTRA_CRNN_BIGRU_VERIFIER_20261016_184000"
COMMAND_BUNDLE = repo_root() / "models" / "macbook" / "AI231_HPO_NOSAMPLER_20261016_210738"


def _model_audio(raw: np.ndarray, capture_rate: int) -> np.ndarray:
    audio = resample(raw, capture_rate, 16000) if capture_rate != 16000 else raw
    return trim_or_pad(normalize_peak(audio), 16000 * 3).astype(np.float32)


def _mapping(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["class"]: row for row in csv.DictReader(fh)}


def _command_text(label: str, mapping: dict[str, dict[str, str]]) -> str:
    if label == "UNKNOWN":
        return "I didn't recognize that command."
    if label == "OUT_OF_SCOPE":
        return "That command is not supported yet."
    if label.endswith("__OTHER_SLOT"):
        return f"{label.removesuffix('__OTHER_SLOT').replace('_', ' ').title()} command recognized; the value needs slot parsing."
    row = mapping.get(label)
    if not row:
        return label
    slot = row["slot_value"] or ""
    return f"Intent: {row['intent']}" + (f" | Slot: {slot}" if slot else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wake-onnx", type=Path, default=WAKE_BUNDLE / "ASTRA_CRNN_BIGRU_VERIFIER_20261016_184000.onnx")
    parser.add_argument("--wake-labels", type=Path, default=WAKE_BUNDLE / "astra_labels.yaml")
    parser.add_argument("--command-onnx", type=Path, default=COMMAND_BUNDLE / "AI231_HPO_NOSAMPLER_20261016_210738.onnx")
    parser.add_argument("--command-labels", type=Path, default=COMMAND_BUNDLE / "ai231_command_variations.yaml")
    parser.add_argument("--command-mapping", type=Path, default=COMMAND_BUNDLE / "benchmark_mapping.csv")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--wake-threshold", type=float, default=0.50)
    parser.add_argument("--wake-confirmations", type=int, default=2)
    parser.add_argument("--wake-silence-rms", type=float, default=0.005)
    parser.add_argument("--command-delay-sec", type=float, default=0.35, help="Brief pause after wake detection before command capture")
    parser.add_argument("--command-duration-sec", type=float, default=3.0)
    parser.add_argument("--command-threshold", type=float, default=0.70, help="Reject uncertain command predictions")
    parser.add_argument("--cooldown-sec", type=float, default=1.5)
    parser.add_argument("--debug-wake", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(argv)

    if args.list_devices:
        for d in list_input_devices():
            print(f"[{d['index']}] {d['name']} ({d['channels']} ch, {d['default_sr']:.0f} Hz)")
        return 0
    required = [args.wake_onnx, args.wake_labels, args.command_onnx, args.command_labels]
    if not all(path.is_file() for path in required):
        parser.error("one or more model-bundle files are missing under models/macbook")
    if args.wake_confirmations < 1 or args.command_duration_sec <= 0 or args.command_delay_sec < 0:
        parser.error("confirmations must be >= 1; command duration > 0; delay >= 0")

    command_labels, _ = load_commands_config(args.command_labels)
    wake = WakeWordInference(args.wake_onnx, args.wake_labels, WAKE_LABEL, args.wake_threshold)
    command = OnnxInference(args.command_onnx, label_map=command_labels)
    command.rejection_threshold = args.command_threshold
    mapping = _mapping(args.command_mapping)
    device = args.device if args.device is not None else default_input_device()

    window = int(args.sample_rate * 3.0)
    hop_sec = 0.5
    command_target = int(args.sample_rate * args.command_duration_sec)
    command_delay = int(args.sample_rate * args.command_delay_sec)
    ring = np.zeros(window, dtype=np.float32)
    write_at = captured = stream_samples = 0
    capture_plan: tuple[int, int] | None = None
    command_parts: list[np.ndarray] = []
    command_ready: np.ndarray | None = None
    lock = threading.Lock()
    overflow_seen = False

    def callback(indata, _frames, _time_info, status):
        nonlocal write_at, captured, stream_samples, capture_plan, command_parts, command_ready, overflow_seen
        chunk = indata[:, 0].copy()
        if status:
            overflow_seen = True
        with lock:
            start, end = stream_samples, stream_samples + len(chunk)
            stream_samples = end
            # Circular wake buffer.
            cursor_end = write_at + len(chunk)
            if cursor_end <= window:
                ring[write_at:cursor_end] = chunk
            else:
                first = window - write_at
                ring[write_at:] = chunk[:first]
                ring[:cursor_end - window] = chunk[first:]
            write_at = cursor_end % window
            captured = min(window, captured + len(chunk))
            # Copy only the planned post-wake command range.
            if capture_plan is not None:
                plan_start, plan_end = capture_plan
                lo, hi = max(start, plan_start), min(end, plan_end)
                if lo < hi:
                    command_parts.append(chunk[lo - start:hi - start])
                if end >= plan_end:
                    command_ready = np.concatenate(command_parts)[:command_target].copy()
                    capture_plan = None

    import sounddevice as sd

    print("Astra Command Controller (Mac)")
    print(f"  mic: device {device}; capture: {args.sample_rate} Hz")
    print(f"  wake: {args.wake_onnx.name} | command: {args.command_onnx.name}")
    print('  [ACTIVE] Say "Astra", then speak a command after the prompt. Ctrl+C stops.')
    positives, cooldown_until, next_wake_eval = 0, 0.0, 0.0
    waiting_for_command = False
    try:
        with sd.InputStream(samplerate=args.sample_rate, channels=1, dtype="float32", device=device,
                            blocksize=max(1, int(args.sample_rate * 0.05)), latency="high", callback=callback):
            while True:
                time.sleep(0.04)
                now = time.perf_counter()
                with lock:
                    ready = captured >= window
                    wake_audio = np.concatenate((ring[write_at:], ring[:write_at])).copy() if ready else None
                    completed_command = command_ready
                    if completed_command is not None:
                        command_ready = None
                    overflow = overflow_seen
                    overflow_seen = False
                if overflow:
                    print("[WARNING] microphone overflow")
                if completed_command is not None:
                    prediction = command.predict(_model_audio(completed_command, args.sample_rate))
                    print(f"[COMMAND] {_command_text(prediction.label, mapping)}")
                    print(f"          class={prediction.label} confidence={prediction.confidence:.1%} latency={prediction.latency_ms:.1f} ms")
                    waiting_for_command = False
                    cooldown_until = time.perf_counter() + args.cooldown_sec
                    continue
                if waiting_for_command or now < cooldown_until or not ready or now < next_wake_eval:
                    continue
                next_wake_eval = now + hop_sec
                if rms(wake_audio) < args.wake_silence_rms:
                    positives = 0
                    continue
                prediction = wake.predict(_model_audio(wake_audio, args.sample_rate))
                score = prediction.raw_probs.get(WAKE_LABEL, 0.0)
                if args.debug_wake:
                    print(f"[DEBUG] wake_score={score:.3f} threshold={wake.threshold:.3f}")
                positives = positives + 1 if wake.is_wake_prediction(prediction) else 0
                if positives >= args.wake_confirmations:
                    with lock:
                        begin = stream_samples + command_delay
                        capture_plan = (begin, begin + command_target)
                        command_parts = []
                    positives = 0
                    waiting_for_command = True
                    print(f"[DETECTED] Astra ({score:.1%}). Speak your command now.")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
