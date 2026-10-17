"""Prepare the raw Astra wake-word corpus and write a loader-compatible manifest.

The source folders stay untouched. Long ambient recordings are converted into
non-overlapping, 3-second, 16 kHz mono WAV windows under ``ambient_segments``.
Synthetic VoiceStudio clips remain training-only; real clips are assigned to
deterministic clip/source-disjoint splits for honest validation and testing.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import wave
from collections import defaultdict
from pathlib import Path


AUDIO_SUFFIXES = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}
MANIFEST_FIELDS = [
    "file", "speaker_id", "intent", "phrase_id", "environment", "device",
    "distance_cm", "split", "source", "synthetic",
]


def audio_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def duration_seconds(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def segment_ambient(raw_dir: Path, out_dir: Path, window_sec: float) -> dict[Path, list[Path]]:
    """Create deterministic full-length ambient windows; never alter sources."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required to create ambient segments")
    result: dict[Path, list[Path]] = {}
    for source in audio_files(raw_dir):
        target_dir = out_dir / source.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        windows: list[Path] = []
        for index in range(int(duration_seconds(source) // window_sec)):
            target = target_dir / f"{source.stem}_{index:03d}.wav"
            expected_frames = round(16000 * window_sec)
            exact = False
            if target.exists():
                try:
                    with wave.open(str(target), "rb") as audio:
                        exact = (
                            audio.getnchannels() == 1
                            and audio.getframerate() == 16000
                            and audio.getnframes() == expected_frames
                        )
                except wave.Error:
                    exact = False
            if not exact:
                subprocess.run(
                    # Place -ss after -i: accurate seek avoids AAC/MP3 packet
                    # boundaries yielding short windows.  -t then gives the
                    # exact fixed-duration model input before resampling.
                    [ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(source), "-ss", str(index * window_sec),
                     "-t", str(window_sec),
                     "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
                    check=True,
                )
            windows.append(target)
        result[source] = windows
    return result


def grouped_splits(groups: dict[str, list[Path]], seed: int) -> dict[Path, str]:
    """Assign complete confuser/ambient-source groups to 70/15/15 splits."""
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    train_end = round(len(keys) * 0.70)
    valid_end = train_end + round(len(keys) * 0.15)
    assignments: dict[Path, str] = {}
    for index, key in enumerate(keys):
        split = "train" if index < train_end else "valid" if index < valid_end else "test"
        assignments.update({path: split for path in groups[key]})
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/wake_word_raw"))
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    root = args.raw.resolve()
    synthetic_root = root / "voicestudio_wake"
    positives_dir = synthetic_root / "ASTRA"
    negatives_dir = synthetic_root / "NOT_WAKE"
    ambient_raw = root / "ambient_noise"
    for directory in (positives_dir, negatives_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required directory is missing: {directory}")
    if ambient_raw.is_dir():
        segments = segment_ambient(ambient_raw, root / "ambient_segments", args.window_sec)
    else:
        # The long original recordings are optional after their fixed 3-second
        # windows have been prepared. This supports moving only portable
        # training assets into a clean repository.
        segment_root = root / "ambient_segments"
        segments = {
            directory: audio_files(directory)
            for directory in sorted(segment_root.iterdir())
            if directory.is_dir() and audio_files(directory)
        } if segment_root.is_dir() else {}
        if not segments:
            raise FileNotFoundError("Need ambient_noise or prepared ambient_segments")
    rows: list[dict[str, str]] = []
    synthetic_paths: set[Path] = set()
    with (synthetic_root / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        for original in csv.DictReader(handle):
            path = synthetic_root / original["path"]
            if not path.is_file():
                raise FileNotFoundError(f"Synthetic manifest points to missing file: {path}")
            synthetic_paths.add(path.resolve())
            rows.append({
                "file": str(path.relative_to(root)), "speaker_id": f"synthetic_{original['voice_id']}",
                "intent": original["intent"], "phrase_id": original["text"],
                "environment": "synthetic", "device": "VoiceStudio OmniVoice", "distance_cm": "",
                "split": "train", "source": "synthetic_voicestudio", "synthetic": "true",
            })

    real_positive = [path for path in audio_files(positives_dir) if path.resolve() not in synthetic_paths]
    positive_groups = {f"positive_{path.stem}": [path] for path in real_positive}
    positive_splits = grouped_splits(positive_groups, args.seed)
    for path in real_positive:
        rows.append({
            "file": str(path.relative_to(root)), "speaker_id": f"clip_recorded_astra_{path.stem}",
            "intent": "ASTRA", "phrase_id": path.stem, "environment": "recorded",
            "device": "user_recording", "distance_cm": "", "split": positive_splits[path],
            "source": "recorded_astra", "synthetic": "false",
        })

    real_negative = [path for path in audio_files(negatives_dir) if path.resolve() not in synthetic_paths]
    negative_groups: dict[str, list[Path]] = defaultdict(list)
    for path in real_negative:
        key = re.sub(r"(?:_?\d+)(?:\s+\d+)?$", "", path.stem).rstrip("_ ")
        negative_groups[f"confuser_{key}"].append(path)
    for source, paths in segments.items():
        negative_groups[f"ambient_{source.stem}"].extend(paths)
    negative_splits = grouped_splits(dict(negative_groups), args.seed + 1)
    for group, paths in sorted(negative_groups.items()):
        for path in sorted(paths):
            is_ambient = group.startswith("ambient_")
            rows.append({
                "file": str(path.relative_to(root)), "speaker_id": f"source_{group}",
                "intent": "NOT_WAKE", "phrase_id": group, "environment": "ambient" if is_ambient else "recorded_confuser",
                "device": "ambient_recording" if is_ambient else "user_recording", "distance_cm": "",
                "split": negative_splits[path], "source": "ambient_segment" if is_ambient else "recorded_not_wake",
                "synthetic": "false",
            })

    rows.sort(key=lambda row: (row["split"], row["intent"], row["file"]))
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["split"], row["intent"])] += 1
    print(f"Wrote {len(rows)} rows to {root / 'metadata.csv'}")
    for key in sorted(counts):
        print(f"{key[0]:5s} {key[1]:8s} {counts[key]}")
    print(f"Ambient windows: {sum(len(paths) for paths in segments.values())} × {args.window_sec:g}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
