"""Raspberry Pi inference entry point.

Runs the exported ONNX model (FP32 or INT8) with onnxruntime on the Pi's CPU,
reads from the Pi's microphone, and drives the GPIO/simulated actions.

Usage::

    python -m app.raspberry_pi.inference --model models/vcm_int8.onnx \
        [--mode vad|push] [--device INDEX]

No PyTorch is required here - only onnxruntime + the shared audio code.
"""
from __future__ import annotations

import argparse
import select
import sys
from pathlib import Path

import numpy as np

# Shared, framework-agnostic pieces (safe to import on the Pi).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.audio.capture import (  # noqa: E402
    default_input_device,
    list_input_devices,
    open_input_stream,
    record_push_to_talk,
    record_vad,
)
from src.audio.features import load_feature_config  # noqa: E402
from src.config import LabelMap, load_commands_config, repo_root  # noqa: E402
from src.inference import OnnxInference  # noqa: E402

from .actions import cleanup, execute  # noqa: E402


def _wait_key(prompt: str) -> str:
    """Block for a single keypress (works with a USB keyboard on the Pi)."""
    print(prompt, end="", flush=True)
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return input()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Raspberry Pi Voice Command inference")
    parser.add_argument("--model", default=str(repo_root() / "models" / "vcm_int8.onnx"))
    parser.add_argument("--mode", default="push", choices=["push", "vad"])
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(argv)

    if args.list_devices:
        for d in list_input_devices():
            print(d)
        return 0

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    label_map, _ = load_commands_config(repo_root() / "configs" / "commands.yaml")
    inference = OnnxInference(model_path, label_map=label_map)
    fc = inference.feature_cfg
    device = args.device if args.device is not None else default_input_device()

    print(f"[rpi] model = {model_path.name}  device = {device}  mode = {args.mode}")
    print("[rpi] Press SPACE to record, Q to quit.\n")

    with open_input_stream(fc.sample_rate, 1, device) as stream:
        while True:
            key = _wait_key("  > ")
            if key.lower() == "q":
                break
            if key != "\x20" and key != " ":
                continue

            print("  Recording...")
            if args.mode == "vad":
                from src.audio.vad import VadConfig

                audio = record_vad(stream, VadConfig(sample_rate=fc.sample_rate))
            else:
                audio = record_push_to_talk(stream, fc.sample_rate, max_duration_sec=3.0)

            if audio is None or len(audio) == 0:
                print("  (no speech detected)")
                continue

            # Standardize the raw capture to the exact training representation.
            from src.audio.preprocessing import normalize_peak, trim_or_pad

            audio = normalize_peak(audio, fc.normalize_target)
            audio = trim_or_pad(audio, int(fc.sample_rate * fc.max_duration_sec))
            pred = inference.predict(audio)
            print(f"  Command: {pred.label}  Confidence: {100 * pred.confidence:.1f}%  "
                  f"Latency: {pred.latency_ms:.1f} ms")
            action = execute(pred.label)
            print(f"  Action:  {action}\n")

    cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
