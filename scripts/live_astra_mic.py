#!/usr/bin/env python3
"""Record from a Mac microphone and run the Astra ONNX wake-word model."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.preprocessing import normalize_peak, resample, trim_or_pad
from src.config import load_commands_config
from src.inference import OnnxInference


BUNDLE = ROOT / "models" / "macbook" / "ASTRA_CRNN_BIGRU_VERIFIER_20261016_184000"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BUNDLE / "ASTRA_CRNN_BIGRU_VERIFIER_20261016_184000.onnx")
    parser.add_argument("--labels-config", type=Path, default=BUNDLE / "astra_labels.yaml")
    parser.add_argument("--device", type=int, default=None, help="Input-device index; default is the Mac system microphone")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Mic capture rate; 48000 is normal for MacBook microphones")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds recorded after Enter")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--save", type=Path, default=None, help="Optional WAV path for the captured audio")
    args = parser.parse_args()

    import sounddevice as sd
    if args.list_devices:
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                print(f"{i}: {device['name']} (default {device['default_samplerate']:.0f} Hz)")
        return 0
    if not args.model.is_file() or not args.labels_config.is_file():
        raise FileNotFoundError("Model bundle is missing. Download it before recording.")

    labels, _ = load_commands_config(args.labels_config)
    engine = OnnxInference(args.model, label_map=labels)
    engine.rejection_threshold = 0.0  # Match the evaluated standard argmax rule.
    input("Press Enter, then say 'Astra' naturally... ")
    print(f"Recording {args.duration:.1f} seconds at {args.sample_rate} Hz...")
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
        print(f"Saved standardized recording: {args.save}")
    prediction = engine.predict(audio)
    print(f"\nResult: {prediction.label}")
    print(f"Confidence: {prediction.confidence:.1%}")
    print(f"Latency: {prediction.latency_ms:.1f} ms")
    print("Probabilities:", ", ".join(f"{name}={prob:.1%}" for name, prob in prediction.raw_probs.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
