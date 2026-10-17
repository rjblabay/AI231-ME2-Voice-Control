#!/usr/bin/env python3
"""Add 160 training-only Astra positives using the local VoiceStudio API."""
from __future__ import annotations

import csv
import json
import time
import urllib.request
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "wake_word_raw" / "voicestudio_wake"
URL = "http://127.0.0.1:3900"
VOICES = [
    ("m01", "male young adult"), ("m02", "male young adult low voice"),
    ("m03", "male middle-aged"), ("m04", "male older adult low voice"),
    ("f01", "female young adult"), ("f02", "female young adult"),
    ("f03", "female middle-aged"), ("f04", "female older adult"),
]
DELIVERIES = [
    "Speak clearly at a natural pace, neutral energy, with a short pause after the word.",
    "Speak slightly slowly and softly, in a calm conversational tone, with a brief pause before the word.",
    "Speak briskly with confident medium energy, natural pitch, then leave a short pause.",
    "Speak the word gently at lower volume with a relaxed pace and natural pause.",
    "Speak with lively medium-high energy, slightly higher pitch, then pause briefly.",
]


def synthesize(instruction: str, seed: int) -> bytes:
    payload = json.dumps({
        "model": "omnivoice", "voice": "default", "input": "Astra",
        "instructions": instruction, "response_format": "wav", "language": "en",
        "seed": seed, "num_step": 16,
    }).encode()
    request = urllib.request.Request(
        f"{URL}/v1/audio/speech", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"VoiceStudio HTTP {response.status}")
        return response.read()


def verify(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getnframes() < 1:
            raise ValueError(f"Invalid WAV: {path}")
        return wav.getframerate(), wav.getnframes() / wav.getframerate()


def main() -> int:
    manifest = OUT / "metadata.csv"
    existing = list(csv.DictReader(manifest.open(encoding="utf-8"))) if manifest.exists() else []
    fields = ["path", "intent", "split", "source", "voice_id", "text", "instruction", "seed", "sample_rate", "duration_sec"]
    additions: list[dict[str, str]] = []
    for voice_index, (voice_id, profile) in enumerate(VOICES):
        for delivery_index, delivery in enumerate(DELIVERIES):
            for repeat in range(4):
                seed = 20_000 + voice_index * 100 + delivery_index * 10 + repeat
                name = f"v2_{voice_id}_{seed}.wav"
                path = OUT / "ASTRA" / name
                instruction = f"{profile}. {delivery}"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(synthesize(instruction, seed))
                    time.sleep(0.12)
                sample_rate, duration = verify(path)
                additions.append({
                    "path": str(path.relative_to(OUT)), "intent": "ASTRA", "split": "train",
                    "source": "synthetic_voicestudio", "voice_id": f"v2_{voice_id}", "text": "Astra",
                    "instruction": instruction, "seed": str(seed), "sample_rate": str(sample_rate),
                    "duration_sec": f"{duration:.3f}",
                })
                print(f"[{len(additions):03d}/160] {voice_id} {name} {duration:.2f}s", flush=True)
    by_path = {row["path"]: row for row in existing}
    by_path.update({row["path"]: row for row in additions})
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(by_path.values(), key=lambda row: row["path"]))
    print(f"Generated {len(additions)} expansion clips; manifest now has {len(by_path)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
