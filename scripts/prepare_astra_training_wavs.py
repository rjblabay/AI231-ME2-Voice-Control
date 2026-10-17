#!/usr/bin/env python3
"""Convert the Astra manifest into portable 16 kHz mono WAV training audio."""
from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "wake_word_raw"
DEST = ROOT / "data" / "processed" / "astra_wake_v1"


def main() -> None:
    rows = list(csv.DictReader((SOURCE / "metadata.csv").open(encoding="utf-8")))
    if not rows:
        raise RuntimeError("No rows in data/wake_word_raw/metadata.csv")
    if DEST.exists():
        shutil.rmtree(DEST)
    audio_dir = DEST / "audio"
    audio_dir.mkdir(parents=True)
    fields = list(rows[0])
    with (DEST / "metadata.csv").open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            source = SOURCE / row["file"]
            if not source.exists():
                raise FileNotFoundError(source)
            cls_dir = audio_dir / row["intent"]
            cls_dir.mkdir(exist_ok=True)
            output = cls_dir / f"{i:04d}_{source.stem}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-ac", "1", "-ar", "16000",
                 "-t", "3", "-c:a", "pcm_s16le", str(output)],
                check=True,
            )
            row["file"] = str(output.relative_to(DEST))
            writer.writerow(row)
    print(f"Prepared {len(rows)} clips -> {DEST}")


if __name__ == "__main__":
    main()
