"""Shared configuration and runtime helpers.

Central place for:
  * loading YAML configs
  * resolving the command taxonomy (labels <-> ids)
  * reproducible random seeding
  * automatic device selection (MPS -> CPU)

Kept dependency-light so it can be imported from anywhere in the project
without pulling in torch.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of the `src` package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """Return the repository root directory."""
    return REPO_ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict. Raises a clear error if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top level of {p}")
    return data


def resolve_config_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve relative paths in a config dict against the repo root.

    Only resolves the well-known path-like keys we use. Non-path values are
    left untouched. This keeps configs portable (no absolute paths).
    """
    def _fix(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _fix(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_fix(x) for x in v]
        if isinstance(v, str) and v and not v.startswith(("/", "~")):
            candidate = REPO_ROOT / v
            # Only treat as a path if it looks like a directory reference.
            return str(candidate)
        return v

    # We deliberately do NOT rewrite every string (would corrupt labels,
    # optimizer names, etc.). Instead we resolve a fixed set of keys.
    resolved = cfg
    for key in ("checkpoint_dir", "results_dir"):
        if key in resolved.get("output", {}):
            resolved["output"][key] = str(REPO_ROOT / resolved["output"][key])
    for ds in resolved.get("data", {}).get("datasets", []):
        if "path" in ds:
            ds["path"] = str(REPO_ROOT / ds["path"])
    return resolved


class LabelMap:
    """Bidirectional mapping between command labels and integer ids.

    Ids are assigned in the order the classes appear in ``commands.yaml``.
    The same ordering is used everywhere (datasets, model head, evaluation,
    demo) so a given id always means the same command.
    """

    def __init__(self, classes: list[str]):
        if not classes:
            raise ValueError("LabelMap requires at least one class")
        self.classes: list[str] = list(classes)
        self.id_to_label: dict[int, str] = {i: c for i, c in enumerate(classes)}
        self.label_to_id: dict[str, int] = {c: i for i, c in enumerate(classes)}

    @classmethod
    def from_config(cls, commands_cfg: dict[str, Any]) -> "LabelMap":
        classes = commands_cfg.get("classes")
        if not classes:
            raise ValueError("commands.yaml must define a non-empty 'classes' list")
        return cls(list(classes))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def to_id(self, label: str) -> int:
        try:
            return self.label_to_id[label]
        except KeyError:
            raise KeyError(
                f"Unknown label {label!r}. Known classes: {self.classes}"
            ) from None

    def to_label(self, idx: int) -> str:
        try:
            return self.id_to_label[idx]
        except KeyError:
            raise KeyError(f"Unknown class id {idx}. Range: 0..{len(self)-1}") from None


def load_commands_config(path: str | Path) -> tuple[LabelMap, float]:
    """Load the command taxonomy and the rejection threshold.

    Returns ``(label_map, rejection_threshold)``.
    """
    cfg = load_yaml(path)
    label_map = LabelMap.from_config(cfg)
    threshold = float(cfg.get("rejection", {}).get("threshold", 0.5))
    return label_map, threshold


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if present) PyTorch for reproducibility.

    Note: some Apple MPS kernels are not bit-for-bit deterministic even with
    a fixed seed. CPU runs are fully deterministic.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def pick_device(preferred: str = "auto") -> "Any":
    """Choose the compute device.

    ``preferred`` may be ``auto``, ``mps``, ``cpu``. With ``auto`` we prefer
    Apple Silicon MPS and fall back to CPU. CUDA is never required.
    """
    import torch

    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        return torch.device("cuda")
    if preferred == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available on this machine.")
        return torch.device("mps")
    return torch.device("cpu")
