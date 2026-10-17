"""Speaker-disjoint train/validation/test splitting.

The core anti-leakage guarantee: whenever speaker information is present, no
speaker may appear in more than one split. A naive random row split would put
the same person's recordings in train AND test, inflating accuracy and
producing a model that does not generalize to new speakers.

Public API:
    * :func:`speaker_disjoint_split` - split a list of (index, speaker_id)
    * :func:`stratified_split` - fallback when no speaker info exists
    * :func:`split_indices` - dispatch on the configured method
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def speaker_disjoint_split(
    speaker_ids: Sequence[str | None],
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices so each speaker lands in exactly one split.

    Args:
        speaker_ids: one speaker id per sample (may contain ``None`` for
            unknown-speaker samples; those are grouped under a single
            ``__nospeaker__`` bucket so they still don't leak).
        val_frac: fraction of *speakers* assigned to validation.
        test_frac: fraction of *speakers* assigned to test.

    Returns:
        (train_idx, val_idx, test_idx) as arrays of positions into the input.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1.0:
        raise ValueError("Require 0 <= val_frac, test_frac and val_frac+test_frac < 1")

    rng = np.random.default_rng(seed)

    # Group sample indices by speaker.
    groups: dict[str, list[int]] = {}
    for i, spk in enumerate(speaker_ids):
        key = spk if spk not in (None, "") else "__nospeaker__"
        groups.setdefault(key, []).append(i)

    speakers = list(groups.keys())
    rng.shuffle(speakers)

    n = len(speakers)
    n_test = max(1, int(round(n * test_frac))) if n > 1 else 0
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    # Guard against degenerate tiny speaker sets.
    if n_test + n_val >= n:
        n_test = max(0, n - 2)
        n_val = 1 if n >= 2 else 0

    test_speakers = set(speakers[:n_test])
    val_speakers = set(speakers[n_test : n_test + n_val])

    train_idx, val_idx, test_idx = [], [], []
    for spk, idxs in groups.items():
        if spk in test_speakers:
            test_idx.extend(idxs)
        elif spk in val_speakers:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    return (
        np.array(sorted(train_idx), dtype=int),
        np.array(sorted(val_idx), dtype=int),
        np.array(sorted(test_idx), dtype=int),
    )


def stratified_split(
    labels: Sequence[int],
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split by label (used only when no speaker info exists).

    Keeps the class distribution similar across splits. NOT speaker-aware -
    use :func:`speaker_disjoint_split` whenever speaker ids are available.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(labels):
        idxs = np.where(np.asarray(labels) == label)[0]
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test : n_test + n_val].tolist())
        train_idx.extend(idxs[n_test + n_val :].tolist())
    return (
        np.array(sorted(train_idx), dtype=int),
        np.array(sorted(val_idx), dtype=int),
        np.array(sorted(test_idx), dtype=int),
    )


def split_indices(
    speaker_ids: Sequence[str | None] | None,
    labels: Sequence[int],
    method: str = "speaker_disjoint",
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the appropriate splitter based on ``method``.

    ``method``:
        * ``speaker_disjoint`` - requires speaker_ids (falls back to
          stratified if speaker_ids is None/empty)
        * ``stratified`` - by label
        * ``official`` - caller supplies pre-split indices; this function is
          not used for that path (handled in datasets.py)
    """
    if method == "speaker_disjoint":
        if speaker_ids is not None and len(speaker_ids) == len(labels):
            return speaker_disjoint_split(speaker_ids, val_frac, test_frac, seed)
        return stratified_split(labels, val_frac, test_frac, seed)
    if method == "stratified":
        return stratified_split(labels, val_frac, test_frac, seed)
    raise ValueError(f"Unknown split method: {method!r}")
