#!/usr/bin/env python3
"""Create training-only synthetic-Astra ablation manifests for HPO.

Validation and test rows are identical in every manifest.  Only the number of
synthetic ASTRA training rows changes, so model-selection remains test-blind.
"""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "astra_wake_v1"
OUT = ROOT / "data" / "hpo"


def main() -> None:
    rows = list(csv.DictReader((SOURCE / "metadata.csv").open(encoding="utf-8")))
    fields = list(rows[0])
    synthetic_astra = [r for r in rows if r["split"] == "train" and r["intent"] == "ASTRA" and r["synthetic"] == "true"]
    keep_always = [r for r in rows if r not in synthetic_astra]
    for count in (80, 160, 240):
        target = OUT / f"astra_syn{count}"
        target.mkdir(parents=True, exist_ok=True)
        selected = keep_always + synthetic_astra[:count]
        selected.sort(key=lambda r: (r["split"], r["intent"], r["file"]))
        with (target / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        # CommandDataset resolves this from the project working directory.
        (target / "source_raw").write_text("data/processed/astra_wake_v1\n", encoding="utf-8")
        print(f"{target}: {len(selected)} rows, {count} synthetic Astra train rows")


if __name__ == "__main__":
    main()
