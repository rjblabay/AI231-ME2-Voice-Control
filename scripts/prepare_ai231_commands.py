"""Prepare the AI231 ME2 Hugging Face command dataset for model training.

This deliberately consumes only the public ``train``, ``test``, and
``synthetic_negatives`` splits.  It never reads the benchmark ``holdout``
split: that remains a final, live Raspberry Pi evaluation.

The output has the repository's standard ``metadata.csv`` format plus a
generated, stable taxonomy.  Each of the 93 official benchmark variations is
one class; this preserves both intent and slot-value information.  A final
``OUT_OF_SCOPE`` class is used for unrelated speech.  Command-shaped audio
with an unsupported slot (for example, a five-minute timer) stays in a
command-specific ``__OTHER_SLOT`` class so the runtime can preserve its intent
and extract the actual slot value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _slug(value: object) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    return text or "NONE"


def class_name(command: object, variation: object, slot_value: object) -> str:
    """Return the stable classifier class for one benchmark command."""
    return f"{_slug(command)}__V{int(variation)}__{_slug(slot_value)}"


def _audio_bytes(cell: Any) -> bytes:
    """Extract encoded WAV/FLAC bytes from Hugging Face's Audio column."""
    if isinstance(cell, dict):
        payload = cell.get("bytes")
        if payload is not None:
            return bytes(payload)
    raise ValueError("Audio cell has no encoded 'bytes' payload; use undecoded Parquet source files")


def _speaker(row: pd.Series, source: str, row_id: int) -> str:
    value = row.get("speaker_id")
    if value is None or pd.isna(value) or not str(value).strip():
        # Keeping all unknown speakers in one group would put every synthetic
        # negative in the same split.  A deterministic per-file identity is a
        # safer fallback when a source has no speaker annotation.
        return f"{source}:unknown:{row_id}"
    return f"{source}:{str(value).strip()}"


def _validation_speakers(rows: list[dict[str, Any]], fraction: float, seed: int) -> set[str]:
    """Select a deterministic speaker-disjoint validation subset of train."""
    speakers = sorted({r["speaker_id"] for r in rows if r["source_split"] == "train"})
    keyed = sorted(
        speakers,
        key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest(),
    )
    n_valid = max(1, round(len(keyed) * fraction)) if len(keyed) > 1 else 0
    return set(keyed[:n_valid])


def _write_audio(payload: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Normalize codec/container to portable PCM WAV while preserving the audio
    # waveform.  The target sample rate is standardized later by the loader.
    import io

    samples, rate = sf.read(io.BytesIO(payload), always_2d=False)
    sf.write(target, samples, rate, subtype="PCM_16")


def _rows_from_parquet(path: Path, source_split: str, mapping: dict[tuple[str, str, str], tuple[str, int]]) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    required = {"audio", "command", "variation", "slot_value", "out_of_scope"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} misses required dataset columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    source_name = f"ai231:{source_split}"
    for rid, row in frame.iterrows():
        is_oos = bool(row["out_of_scope"])
        if is_oos:
            intent = "OUT_OF_SCOPE"
            command, variation, slot = "OUT_OF_SCOPE", "", ""
        else:
            command = str(row["command"]).strip()
            # The dataset's ``variation`` field is the canonical phrase
            # (e.g. "Start music"), not the numeric 1..3 benchmark id.
            variation_phrase = str(row["variation"]).strip()
            slot = "" if pd.isna(row["slot_value"]) else str(row["slot_value"]).strip()
            key = (command, variation_phrase, slot)
            if key not in mapping:
                # The dataset deliberately includes command-shaped recordings
                # with unsupported slot values (such as a five-minute timer).
                # Keep its intent rather than misclassifying it as an official
                # benchmark value or as a reminder.  A later slot parser can
                # extract e.g. "five minutes" for a TIMER__OTHER_SLOT result.
                intent = f"{_slug(command)}__OTHER_SLOT"
                variation = ""
            else:
                intent, variation = mapping[key]
        rows.append(
            {
                "payload": _audio_bytes(row["audio"]),
                "intent": intent,
                "speaker_id": _speaker(row, source_name, int(rid)),
                "source_split": source_split,
                "command": command,
                "variation": variation,
                "slot_value": slot,
                "transcript": "" if pd.isna(row.get("transcript")) else str(row.get("transcript")),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/ai231_me2_voice_commands")
    parser.add_argument("--output-dir", default="data/processed/ai231_commands_v1")
    parser.add_argument("--labels-output", default="configs/ai231_command_variations.yaml")
    parser.add_argument("--val-speaker-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--force", action="store_true", help="Replace an existing prepared output directory")
    args = parser.parse_args()

    raw_dir = (REPO_ROOT / args.raw_dir).resolve()
    out_dir = (REPO_ROOT / args.output_dir).resolve()
    labels_out = (REPO_ROOT / args.labels_output).resolve()
    required_files = [
        raw_dir / "variations.csv",
        raw_dir / "data/train-00000-of-00002.parquet",
        raw_dir / "data/train-00001-of-00002.parquet",
        raw_dir / "data/test-00000-of-00001.parquet",
        raw_dir / "synthetic_negatives/train-00000-of-00001.parquet",
        raw_dir / "synthetic_negatives/test-00000-of-00001.parquet",
    ]
    absent = [str(p) for p in required_files if not p.exists()]
    if absent:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(absent))
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} already exists; pass --force to replace it")
        shutil.rmtree(out_dir)

    variations = pd.read_csv(raw_dir / "variations.csv").fillna("")
    # The Parquet rows store the phrase string rather than the numeric
    # variation id, so map by (intent, canonical phrase, slot).
    mapping: dict[tuple[str, str, str], tuple[str, int]] = {}
    mapping_rows: list[dict[str, str]] = []
    for _, row in variations.iterrows():
        command, variation, value = str(row["label"]), int(row["variation"]), str(row["value"])
        phrase = str(row["phrase"])
        key = (command, phrase, value)
        label = class_name(command, variation, value)
        if key in mapping:
            raise ValueError(f"Duplicate benchmark variation key: {key}")
        mapping[key] = (label, variation)
        mapping_rows.append({"class": label, "intent": command, "variation": str(variation), "slot_value": value, "phrase": str(row["phrase"])})
    if len(mapping) != 93:
        raise ValueError(f"Expected 93 official variations, found {len(mapping)}")

    records: list[dict[str, Any]] = []
    for path in sorted((raw_dir / "data").glob("train-*.parquet")):
        records.extend(_rows_from_parquet(path, "train", mapping))
    records.extend(_rows_from_parquet(raw_dir / "synthetic_negatives/train-00000-of-00001.parquet", "train", mapping))
    records.extend(_rows_from_parquet(raw_dir / "data/test-00000-of-00001.parquet", "test", mapping))
    records.extend(_rows_from_parquet(raw_dir / "synthetic_negatives/test-00000-of-00001.parquet", "test", mapping))

    # The data card guarantees its official test speakers are separate.  The
    # validation speakers are selected only from official train.
    valid_speakers = _validation_speakers(records, args.val_speaker_frac, args.seed)
    rows: list[dict[str, str]] = []
    for index, record in enumerate(records):
        split = record["source_split"]
        if split == "train" and record["speaker_id"] in valid_speakers:
            split = "valid"
        rel = Path("audio") / split / f"{index:06d}.wav"
        _write_audio(record.pop("payload"), out_dir / rel)
        rows.append(
            {
                "file": rel.as_posix(), "intent": record["intent"], "speaker_id": record["speaker_id"],
                "split": split, "command": str(record["command"]), "variation": str(record["variation"]),
                "slot_value": str(record["slot_value"]), "transcript": record["transcript"],
            }
        )
        if (index + 1) % 500 == 0:
            print(f"[prepare] wrote {index + 1}/{len(records)} clips")

    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (out_dir / "benchmark_mapping.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(mapping_rows[0]))
        writer.writeheader(); writer.writerows(mapping_rows)
    extended_classes = sorted({r["intent"] for r in rows if r["intent"].endswith("__OTHER_SLOT")})
    labels = [r["class"] for r in mapping_rows] + extended_classes + ["OUT_OF_SCOPE"]
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.write_text(yaml.safe_dump({"rejection": {"threshold": 0.70}, "classes": labels}, sort_keys=False), encoding="utf-8")
    summary = {"records": len(rows), "classes": len(labels), "extended_slot_classes": extended_classes, "splits": dict(Counter(r["split"] for r in rows)), "labels_config": str(labels_out.relative_to(REPO_ROOT)), "holdout_used": False}
    (out_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[prepare] " + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
