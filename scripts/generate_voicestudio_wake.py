"""Generate a training-only synthetic Astra wake-word augmentation set.

The script uses a locally running VoiceStudio backend.  It deliberately writes
only ``split=train`` records: synthetic speech must never appear in validation
or test evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
import wave
from pathlib import Path


VOICES = [
    ("m01", "male, young adult, moderate pitch, american accent"),
    ("m02", "male, young adult, low pitch, british accent"),
    ("m03", "male, middle-aged, moderate pitch, indian accent"),
    ("m04", "male, elderly, low pitch, australian accent"),
    ("f01", "female, young adult, high pitch, american accent"),
    ("f02", "female, young adult, moderate pitch, british accent"),
    ("f03", "female, middle-aged, moderate pitch, canadian accent"),
    ("f04", "female, elderly, low pitch, indian accent"),
]

NEGATIVES = [
    "aster",
    "astra nova",
    "asteroid",
    "ask her",
    "asra",
    "hello there",
    "turn on the lights",
    "what is the weather",
    "play some music",
    "nothing",
]


def synthesize(url: str, text: str, instruction: str, seed: int) -> bytes:
    body = json.dumps(
        {
            "model": "omnivoice",
            "voice": "default",
            "input": text,
            "instructions": instruction,
            "response_format": "wav",
            "language": "en",
            "seed": seed,
            "num_step": 16,
        }
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"VoiceStudio returned HTTP {response.status}")
        return response.read()


def verify_wav(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as audio:
        frames = audio.getnframes()
        rate = audio.getframerate()
        if audio.getnchannels() != 1 or rate <= 0 or frames <= 0:
            raise ValueError(f"Invalid generated WAV: {path}")
        return rate, frames / rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3900")
    parser.add_argument("--out", type=Path, default=Path("data/raw/voicestudio_wake"))
    parser.add_argument("--count-per-voice", type=int, default=10)
    parser.add_argument("--pause-sec", type=float, default=0.15)
    args = parser.parse_args()
    if args.count_per_voice != len(NEGATIVES):
        raise ValueError(f"count-per-voice must be {len(NEGATIVES)} for a balanced manifest")

    out = args.out.resolve()
    manifest_path = out / "metadata.csv"
    rows: list[dict[str, str]] = []
    planned: list[tuple[str, str, str, str, int]] = []
    for voice_index, (voice_id, instruction) in enumerate(VOICES):
        for item_index in range(args.count_per_voice):
            seed = 10_000 + voice_index * 100 + item_index
            planned.append(("ASTRA", voice_id, instruction, "Astra", seed))
            planned.append(("NOT_WAKE", voice_id, instruction, NEGATIVES[item_index], seed + 50_000))

    for label, voice_id, instruction, text, seed in planned:
        basename = f"{voice_id}_{seed:05d}.wav"
        path = out / label / basename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            payload = synthesize(args.url, text, instruction, seed)
            path.write_bytes(payload)
            time.sleep(args.pause_sec)
        sample_rate, duration = verify_wav(path)
        rows.append(
            {
                "path": str(path.relative_to(out)),
                "intent": label,
                "split": "train",
                "source": "synthetic_voicestudio",
                "voice_id": voice_id,
                "text": text,
                "instruction": instruction,
                "seed": str(seed),
                "sample_rate": str(sample_rate),
                "duration_sec": f"{duration:.3f}",
            }
        )
        print(f"[{len(rows):03d}/{len(planned):03d}] {label:<8} {path.name} ({duration:.2f}s)", flush=True)

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} clips and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
