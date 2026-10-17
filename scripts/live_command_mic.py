#!/usr/bin/env python3
"""Record a spoken command with a Mac microphone and run the command ONNX model.

Example:
    .venv/bin/python scripts/live_command_mic.py --device 2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.preprocessing import normalize_peak, resample, trim_or_pad
from src.config import load_commands_config
from src.inference import OnnxInference


BUNDLE = ROOT / "models" / "macbook" / "AI231_HPO_NOSAMPLER_20261016_210738"


def _mapping(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["class"]: row for row in csv.DictReader(fh)}


def _display(label: str, mapping: dict[str, dict[str, str]]) -> str:
    if label.endswith("__OTHER_SLOT"):
        return f"intent={label.removesuffix('__OTHER_SLOT')} | slot=other (duration/value parsing comes next)"
    if label == "OUT_OF_SCOPE":
        return "out of scope / unsupported command"
    row = mapping.get(label)
    if not row:
        return label
    value = row["slot_value"] or "—"
    return f"intent={row['intent']} | slot={value} | phrase={row['phrase']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BUNDLE / "AI231_HPO_NOSAMPLER_20261016_210738.onnx")
    parser.add_argument("--labels-config", type=Path, default=BUNDLE / "ai231_command_variations.yaml")
    parser.add_argument("--mapping", type=Path, default=BUNDLE / "benchmark_mapping.csv")
    parser.add_argument("--device", type=int, default=None, help="Input-device index; default is the Mac system microphone")
    parser.add_argument("--sample-rate", type=int, default=48000, help="MacBook microphones normally capture at 48 kHz")
    parser.add_argument("--duration", type=float, default=3.0, help="Recording duration after Enter")
    parser.add_argument("--threshold", type=float, default=0.0, help="0 = evaluated argmax behavior; raise later after OOS threshold tuning")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the standardized 16 kHz WAV")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    import sounddevice as sd

    if args.list_devices:
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                print(f"{i}: {device['name']} ({device['default_samplerate']:.0f} Hz)")
        return 0
    if not args.model.is_file() or not args.labels_config.is_file():
        raise FileNotFoundError("Command-model bundle is missing. Check models/macbook/.")

    labels, _ = load_commands_config(args.labels_config)
    engine = OnnxInference(args.model, label_map=labels)
    engine.rejection_threshold = args.threshold
    mapping = _mapping(args.mapping)
    print(f"Command model: {args.model.name}")
    print(f"Microphone: device {args.device if args.device is not None else 'system default'}; capture {args.sample_rate} Hz")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            input("Press Enter, then say one command naturally... ")
            print(f"Recording {args.duration:.1f} seconds...")
            raw = sd.rec(int(args.duration * args.sample_rate), samplerate=args.sample_rate,
                         channels=1, dtype="float32", device=args.device)
            sd.wait()
            audio = raw[:, 0]
            audio = resample(audio, args.sample_rate, 16000)
            audio = trim_or_pad(normalize_peak(audio), 16000 * 3).astype(np.float32)
            if args.save:
                import soundfile as sf
                args.save.parent.mkdir(parents=True, exist_ok=True)
                sf.write(args.save, audio, 16000)
                print(f"Saved: {args.save}")
            prediction = engine.predict(audio)
            ranked = sorted(prediction.raw_probs.items(), key=lambda item: item[1], reverse=True)[:3]
            print(f"\nRecognized: {_display(prediction.label, mapping)}")
            print(f"Class: {prediction.label} | confidence: {prediction.confidence:.1%} | latency: {prediction.latency_ms:.1f} ms")
            print("Top 3:", ", ".join(f"{label}={prob:.1%}" for label, prob in ranked), "\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
