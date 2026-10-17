"""Dataset loading for the Voice Command Model.

A "prepared" dataset is a directory containing a ``metadata.csv`` with one row
per audio clip::

    file,speaker_id,intent,phrase_id,environment,device,distance_cm

where ``file`` is a path relative to the dataset directory. This is the format
produced by the ``scripts/prepare_*.py`` tools for all three supported sources
(Fluent Speech Commands, Google Speech Commands V2, and custom ME2
recordings), so the loader does not need to know which source produced the
files.

The loader:
    * reads metadata.csv
    * filters to intents present in the configured taxonomy
    * builds speaker-disjoint / stratified splits
    * yields standardized audio + features on demand

It never assumes all three datasets exist - you enable only the ones you have
in the config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from ..audio.augmentation import augment
from ..audio.features import FeatureConfig, features_from_audio
from ..audio.preprocessing import load_and_standardize
from ..config import LabelMap
from .split import split_indices


@dataclass
class Sample:
    path: Path
    intent: str
    speaker_id: str | None
    phrase_id: str | None = None
    environment: str | None = None
    device: str | None = None
    distance_cm: float | None = None
    split: str | None = None  # official split label (train/valid/test) if provided


@dataclass
class DatasetConfig:
    datasets: list[dict[str, Any]] = field(default_factory=list)
    split_method: str = "speaker_disjoint"
    val_frac: float = 0.15
    test_frac: float = 0.20
    seed: int = 42
    augment: bool = True


class CommandDataset:
    """Loads a prepared dataset and exposes split, iterable tensors.

    Usage::

        ds = CommandDataset(label_map, [ds_cfg], feature_cfg, aug_cfg)
        ds.prepare()                       # loads metadata + computes splits
        for X, y in ds.iter("train"):      # yields (features, label_id)
            ...
    """

    REQUIRED_COLUMNS = {"file", "intent"}

    def __init__(
        self,
        label_map: LabelMap,
        dataset_configs: list[dict[str, Any]],
        feature_cfg: FeatureConfig,
        augment_cfg: dict[str, Any] | None = None,
        split_cfg: dict[str, Any] | None = None,
    ):
        self.label_map = label_map
        self.dataset_configs = dataset_configs
        self.feature_cfg = feature_cfg
        self.augment_cfg = augment_cfg or {}
        split_cfg = split_cfg or {}
        self.split_method = split_cfg.get("method", "speaker_disjoint")
        self.val_frac = float(split_cfg.get("val_frac", 0.15))
        self.test_frac = float(split_cfg.get("test_frac", 0.20))
        self.seed = int(split_cfg.get("seed", 42))
        cache = split_cfg.get("feature_cache", {}) or {}
        self.cache_features = bool(cache.get("enabled", False))
        self.cache_dir = Path(cache.get("path", "data/cache/logmel"))
        if self.cache_features:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples: list[Sample] = []
        self.train_idx: np.ndarray = np.array([], dtype=int)
        self.val_idx: np.ndarray = np.array([], dtype=int)
        self.test_idx: np.ndarray = np.array([], dtype=int)
        self._rng = np.random.default_rng(self.seed)
        self._prepared = False

    # ------------------------------------------------------------------ load
    def _load_one(self, ds_cfg: dict[str, Any]) -> list[Sample]:
        root = Path(ds_cfg["path"])
        meta_path = root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Dataset '{ds_cfg.get('name', '?')}' is missing "
                f"{meta_path}. Run the matching scripts/prepare_*.py first."
            )
        # Some preparers keep the audio in a separate raw tree and write a
        # 'source_raw' pointer file. Resolve relative 'file' paths against it.
        audio_root = root
        ptr = root / "source_raw"
        if ptr.exists():
            audio_root = Path(ptr.read_text(encoding="utf-8").strip())

        df = pd.read_csv(meta_path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{meta_path} is missing required columns: {sorted(missing)}"
            )

        valid_labels = set(self.label_map.classes)
        out: list[Sample] = []
        for _, row in df.iterrows():
            intent = str(row["intent"]).strip()
            if intent not in valid_labels:
                continue  # skip classes not in the taxonomy
            rel = str(row["file"]).strip()
            if Path(rel).is_absolute():
                audio_path = Path(rel)
            else:
                audio_path = audio_root / rel
            if not audio_path.exists():
                continue  # skip missing audio files silently
            spk = row.get("speaker_id")
            spk = None if (spk is None or (isinstance(spk, float) and np.isnan(spk))) else str(spk)
            split_val = row.get("split")
            split_val = None if (split_val is None or (isinstance(split_val, float) and np.isnan(split_val))) else str(split_val).strip().lower()
            out.append(
                Sample(
                    path=audio_path,
                    intent=intent,
                    speaker_id=spk,
                    phrase_id=_opt(row.get("phrase_id")),
                    environment=_opt(row.get("environment")),
                    device=_opt(row.get("device")),
                    distance_cm=_opt_float(row.get("distance_cm")),
                    split=split_val,
                )
            )
        return out

    def prepare(self) -> "CommandDataset":
        """Load all configured datasets and compute the splits."""
        self.samples = []
        for ds_cfg in self.dataset_configs:
            loaded = self._load_one(ds_cfg)
            self.samples.extend(loaded)
            print(f"[data] {ds_cfg.get('name', '?')}: {len(loaded)} usable clips")

        if not self.samples:
            raise RuntimeError(
                "No usable clips found. Check your dataset paths and that "
                "metadata.csv 'intent' values match configs/commands.yaml."
            )

        labels = [self.label_map.to_id(s.intent) for s in self.samples]
        speakers = [s.speaker_id for s in self.samples]

        if self.split_method == "official":
            # Preserve official splits encoded as a 'split' column where present.
            # Datasets WITHOUT a split column (e.g. Speech Commands V2) are
            # assigned speaker-disjoint splits themselves, then the two are
            # merged. This lets FSC (official) + SCV2 (computed) train together
            # without silently dropping the SCV2 clips.
            official = self._official_splits()
            if official is None:
                print("[data] WARNING: split_method='official' but no 'split' "
                      "column found; falling back to speaker_disjoint.")
                self.train_idx, self.val_idx, self.test_idx = split_indices(
                    speakers, labels, "speaker_disjoint", self.val_frac, self.test_frac, self.seed
                )
            else:
                self._merge_with_computed_split(official, speakers, labels)
        else:
            self.train_idx, self.val_idx, self.test_idx = split_indices(
                speakers, labels, self.split_method, self.val_frac, self.test_frac, self.seed
            )

        n_spk_train = len({self.samples[i].speaker_id for i in self.train_idx})
        n_spk_val = len({self.samples[i].speaker_id for i in self.val_idx})
        n_spk_test = len({self.samples[i].speaker_id for i in self.test_idx})
        print(
            f"[data] split: train={len(self.train_idx)} val={len(self.val_idx)} "
            f"test={len(self.test_idx)} | speakers train/val/test = "
            f"{n_spk_train}/{n_spk_val}/{n_spk_test}"
        )
        self._prepared = True
        return self

    def _official_splits(self):
        """Build split index arrays from a 'split' column in metadata.csv.

        Returns ``(train_idx, val_idx, test_idx)`` covering ONLY the samples
        that carry a usable 'split' value, or ``None`` if no sample has one
        (caller falls back to speaker_disjoint). Samples without a 'split'
        value are left out here so the caller can assign them separately -
        this is what lets a dataset with official splits (FSC) be merged with
        one that has none (Speech Commands V2).

        Raises ``ValueError`` if a speaker appears in more than one split -
        that would be data leakage and defeats the purpose of preserving the
        official split.
        """
        norm = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
        buckets: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
        seen_any = False
        for i, s in enumerate(self.samples):
            if not s.split:
                continue
            key = norm.get(s.split)
            if key is None:
                raise ValueError(f"Unknown official split label: {s.split!r}")
            buckets[key].append(i)
            seen_any = True
        if not seen_any:
            return None

        # Leakage guard: no speaker may span two splits.
        spk_to_splits: dict[str, set[str]] = {}
        for split_name, idxs in buckets.items():
            for i in idxs:
                spk = self.samples[i].speaker_id or "__nospeaker__"
                spk_to_splits.setdefault(spk, set()).add(split_name)
        leaks = {spk: ss for spk, ss in spk_to_splits.items() if len(ss) > 1}
        if leaks:
            example = next(iter(leaks))
            raise ValueError(
                f"Official split is NOT speaker-disjoint: speaker {example!r} "
                f"appears in {sorted(leaks[example])}. Refusing to preserve it. "
                "Use split_method='speaker_disjoint' instead."
            )

        return (
            np.array(sorted(buckets["train"]), dtype=int),
            np.array(sorted(buckets["valid"]), dtype=int),
            np.array(sorted(buckets["test"]), dtype=int),
        )

    def _merge_with_computed_split(
        self,
        official: tuple[np.ndarray, np.ndarray, np.ndarray],
        speakers: list[str | None],
        labels: list[int],
    ) -> None:
        """Combine official splits with speaker-disjoint splits for the rest.

        ``official`` assigns splits to the samples that had a 'split' column.
        The remaining samples (datasets without official splits) are assigned
        speaker-disjoint splits, constrained so they never collide with a
        speaker already placed by the official split. The union is then
        verified to be globally speaker-disjoint.
        """
        o_train, o_valid, o_test = official
        assigned = {int(i) for i in list(o_train) + list(o_valid) + list(o_test)}
        unassigned = [i for i in range(len(self.samples)) if i not in assigned]

        # Speakers already locked into an official split.
        locked: set[str] = set()
        for idxs in (o_train, o_valid, o_test):
            for i in idxs:
                locked.add(speakers[i] or "__nospeaker__")

        if unassigned:
            u_speakers = [speakers[i] for i in unassigned]
            u_labels = [labels[i] for i in unassigned]
            # Speaker-disjoint split of just the unassigned samples.
            u_train, u_valid, u_test = split_indices(
                u_speakers, u_labels, "speaker_disjoint", self.val_frac, self.test_frac, self.seed
            )
            # A speaker that ALSO appears in the official split must not land in
            # two splits. Push every such collision into train (the largest
            # split, so it is the least likely to create a new imbalance).
            buckets = {"train": [], "valid": [], "test": []}
            u_train_set = set(u_train.tolist())
            u_valid_set = set(u_valid.tolist())
            for j, i in enumerate(unassigned):
                if (u_speakers[j] or "__nospeaker__") in locked:
                    buckets["train"].append(i)
                elif j in u_train_set:
                    buckets["train"].append(i)
                elif j in u_valid_set:
                    buckets["valid"].append(i)
                else:
                    buckets["test"].append(i)
            o_train = np.concatenate([o_train, np.array(sorted(buckets["train"]), dtype=int)])
            o_valid = np.concatenate([o_valid, np.array(sorted(buckets["valid"]), dtype=int)])
            o_test = np.concatenate([o_test, np.array(sorted(buckets["test"]), dtype=int)])

        # Global leakage guard across BOTH official and computed parts.
        spk_to_splits: dict[str, set[str]] = {}
        for split_name, idxs in (("train", o_train), ("valid", o_valid), ("test", o_test)):
            for i in idxs:
                spk_to_splits.setdefault(speakers[i] or "__nospeaker__", set()).add(split_name)
        leaks = {spk: ss for spk, ss in spk_to_splits.items() if len(ss) > 1}
        if leaks:
            example = next(iter(leaks))
            raise ValueError(
                f"Merged split is NOT speaker-disjoint: speaker {example!r} "
                f"appears in {sorted(leaks[example])}. Use "
                "split_method='speaker_disjoint' to force a clean re-split."
            )

        self.train_idx = np.array(sorted(o_train), dtype=int)
        self.val_idx = np.array(sorted(o_valid), dtype=int)
        self.test_idx = np.array(sorted(o_test), dtype=int)

    # --------------------------------------------------------------- iterate
    def features(self, index: int) -> np.ndarray:
        """Compute standardized features for the sample at global ``index``.

        No augmentation is applied here - augmentation is handled separately
        by :meth:`augmented_features` so validation/test data stays clean.
        """
        cached = self._load_cached_features(index)
        if cached is not None:
            return cached
        s = self.samples[index]
        audio = load_and_standardize(
            s.path,
            sample_rate=self.feature_cfg.sample_rate,
            max_duration_sec=self.feature_cfg.max_duration_sec,
            normalize_target=self.feature_cfg.normalize_target,
        )
        feats = features_from_audio(audio, self.feature_cfg, kind="logmel")
        self._save_cached_features(index, feats)
        return feats

    def augmented_features(self, index: int, feature_kind: str = "logmel", n_features: int | None = None) -> np.ndarray:
        """Features for the train split with a random augmentation applied."""
        # Cached base features make HPC training I/O-bound no longer.  This is
        # only sound when waveform augmentation is disabled; callers that
        # enable augmentation continue to compute it freshly.
        if self.cache_features and not self.augment_cfg.get("enabled", False):
            return self.features(index)
        s = self.samples[index]
        audio = load_and_standardize(
            s.path,
            sample_rate=self.feature_cfg.sample_rate,
            max_duration_sec=self.feature_cfg.max_duration_sec,
            normalize_target=self.feature_cfg.normalize_target,
        )
        if self.augment_cfg.get("enabled", False):
            audio = augment(audio, self.augment_cfg, self.feature_cfg.sample_rate, self._rng)
        return features_from_audio(audio, self.feature_cfg, kind=feature_kind, n_features=n_features)

    def _cache_path(self, index: int) -> Path:
        s = self.samples[index]
        signature = f"{s.path.resolve()}|{self.feature_cfg}|logmel"
        return self.cache_dir / (hashlib.sha256(signature.encode()).hexdigest() + ".npy")

    def _load_cached_features(self, index: int) -> np.ndarray | None:
        if not self.cache_features:
            return None
        path = self._cache_path(index)
        if not path.exists():
            return None
        try:
            return np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        except (OSError, ValueError):
            return None

    def _save_cached_features(self, index: int, feats: np.ndarray) -> None:
        if not self.cache_features:
            return
        path = self._cache_path(index)
        if path.exists():
            return
        tmp = path.with_suffix(".tmp.npy")
        try:
            np.save(tmp, feats.astype(np.float32, copy=False), allow_pickle=False)
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)

    def iter(
        self,
        split: str,
        feature_kind: str = "logmel",
        n_features: int | None = None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Yield (features, label_id) for one split.

        Augmentation is applied ONLY for the ``train`` split.
        """
        if not self._prepared:
            raise RuntimeError("Call prepare() before iterating.")
        idx = {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[split]
        do_aug = split == "train" and self.augment_cfg.get("enabled", False)
        for i in idx:
            if do_aug:
                feats = self.augmented_features(i, feature_kind, n_features)
            else:
                feats = self.features(i)
            yield feats, self.label_map.to_id(self.samples[i].intent)

    def sizes(self) -> dict[str, int]:
        return {
            "train": int(len(self.train_idx)),
            "val": int(len(self.val_idx)),
            "test": int(len(self.test_idx)),
        }


def _opt(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(v)


def _opt_float(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
