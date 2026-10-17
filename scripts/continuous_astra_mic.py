#!/usr/bin/env python3
"""Continuously listen on a Mac microphone for the Astra wake word.

Ctrl-C stops the listener.  The model sees a rolling 3-second window every
``--hop`` seconds; consecutive confirmations suppress one-window false wakes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.preprocessing import normalize_peak, resample
from src.config import load_commands_config
from src.inference import OnnxInference

BUNDLE = ROOT / "models" / "macbook" / "ASTRA_CRNN_BIGRU_VERIFIER_20261016_184000"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BUNDLE / "ASTRA_CRNN_BIGRU_VERIFIER_20261016_184000.onnx")
    parser.add_argument("--labels-config", type=Path, default=BUNDLE / "astra_labels.yaml")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--hop", type=float, default=0.75, help="Seconds between model evaluations")
    parser.add_argument("--astra-threshold", type=float, default=0.50)
    parser.add_argument("--confirmations", type=int, default=2, help="Consecutive positive windows required")
    parser.add_argument("--cooldown", type=float, default=3.0, help="Seconds to suppress repeat wake events")
    parser.add_argument("--verbose", action="store_true", help="Print every Astra probability")
    args = parser.parse_args()

    import sounddevice as sd
    labels, _ = load_commands_config(args.labels_config)
    engine = OnnxInference(args.model, label_map=labels)
    engine.rejection_threshold = 0.0
    window_samples = int(args.sample_rate * 3.0)
    block_samples = int(args.sample_rate * 0.10)
    ring = np.zeros(window_samples, dtype=np.float32)
    filled = 0
    confirmations = 0
    last_wake = -float("inf")
    next_infer = time.monotonic()

    print("Listening for 'Astra' continuously. Press Ctrl-C to stop.")
    try:
        with sd.InputStream(samplerate=args.sample_rate, channels=1, dtype="float32", device=args.device) as stream:
            while True:
                block, overflowed = stream.read(block_samples)
                if overflowed:
                    print("[warning] microphone buffer overflow; continuing")
                block = block[:, 0]
                ring = np.roll(ring, -len(block))
                ring[-len(block):] = block
                filled = min(window_samples, filled + len(block))
                now = time.monotonic()
                if filled < window_samples or now < next_infer:
                    continue
                next_infer = now + args.hop
                audio16 = resample(ring, args.sample_rate, 16000)
                audio16 = normalize_peak(audio16).astype(np.float32)
                prediction = engine.predict(audio16)
                astra_p = prediction.raw_probs.get("ASTRA", 0.0)
                positive = astra_p >= args.astra_threshold
                confirmations = confirmations + 1 if positive else 0
                if args.verbose:
                    print(f"Astra={astra_p:.1%} confirmations={confirmations}/{args.confirmations}")
                if confirmations >= args.confirmations and now - last_wake >= args.cooldown:
                    print(f"\nWAKE DETECTED — Astra confidence {astra_p:.1%}\n")
                    last_wake = now
                    confirmations = 0
    except KeyboardInterrupt:
        print("\nListener stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
