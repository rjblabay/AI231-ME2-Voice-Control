"""Training CLI for the Voice Command Model.

Usage::

    python -m src.train --config configs/dscnn.yaml

Loads the configured datasets, builds the model, trains with early stopping,
and saves the best checkpoint plus training history and metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score

from .audio.augmentation import load_augment_config
from .audio.features import load_feature_config
from .config import (
    load_commands_config,
    load_yaml,
    pick_device,
    repo_root,
    resolve_config_paths,
    set_seed,
)
from .data.datasets import CommandDataset
from .models.factory import build_model, count_parameters


class _TrainSet(Dataset):
    """Training split with per-sample random augmentation applied on access."""

    def __init__(
        self,
        cmd_ds: CommandDataset,
        feature_kind: str,
        n_features: int | None,
        feature_augment_cfg: dict[str, Any] | None = None,
    ):
        self.ds = cmd_ds
        self.indices = cmd_ds.train_idx.tolist()
        self.labels = [cmd_ds.label_map.to_id(cmd_ds.samples[i].intent) for i in self.indices]
        self.feature_kind = feature_kind
        self.n_features = n_features
        # Feature masking is intentionally applied after reading the cached
        # deterministic log-Mel. Unlike waveform augmentation it therefore
        # keeps the HPC cache useful while regularising only training data.
        self.feature_augment_cfg = feature_augment_cfg or {}
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        feats = self.ds.augmented_features(self.indices[i], self.feature_kind, self.n_features)
        if self.feature_augment_cfg.get("enabled", False):
            feats = self._feature_augment(feats)
        return torch.from_numpy(feats), self.labels[i]

    def _feature_augment(self, feats: np.ndarray) -> np.ndarray:
        """Apply lightweight SpecAugment-style masking to a log-Mel tensor."""
        cfg = self.feature_augment_cfg
        if self._rng.random() > float(cfg.get("p", 1.0)):
            return feats
        out = feats.copy()
        time_frames, mel_bins = out.shape
        for _ in range(int(cfg.get("time_masks", 0))):
            width = int(self._rng.integers(0, min(int(cfg.get("time_mask_max_frames", 0)), time_frames) + 1))
            if width:
                start = int(self._rng.integers(0, time_frames - width + 1))
                out[start : start + width, :] = 0.0
        for _ in range(int(cfg.get("freq_masks", 0))):
            width = int(self._rng.integers(0, min(int(cfg.get("freq_mask_max_bins", 0)), mel_bins) + 1))
            if width:
                start = int(self._rng.integers(0, mel_bins - width + 1))
                out[:, start : start + width] = 0.0
        return out


class _EvalSet(Dataset):
    """Validation/test split - clean features, no augmentation."""

    def __init__(self, cmd_ds: CommandDataset, split: str, feature_kind: str, n_features: int | None):
        self.ds = cmd_ds
        self.indices = {"val": cmd_ds.val_idx, "test": cmd_ds.test_idx}[split].tolist()
        self.labels = [cmd_ds.label_map.to_id(cmd_ds.samples[i].intent) for i in self.indices]
        self.feature_kind = feature_kind
        self.n_features = n_features

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        feats = self.ds.features(self.indices[i])
        return torch.from_numpy(feats), self.labels[i]


def _class_weights(cmd_ds: CommandDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights computed from the training split."""
    counts = np.zeros(num_classes, dtype=float)
    for i in cmd_ds.train_idx:
        counts[cmd_ds.label_map.to_id(cmd_ds.samples[i].intent)] += 1
    present = counts[counts > 0]
    if present.size == 0:
        return torch.ones(num_classes)
    w = present.sum() / (num_classes * counts + 1e-8)
    return torch.tensor(w, dtype=torch.float32)


def _epoch(model, loader, criterion, optimizer, device, collect_preds=False, gradient_clip_norm=None):
    is_train = optimizer is not None
    model.train(is_train)
    total, n = 0.0, 0
    preds, trues = [], []
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            if collect_preds:
                preds.append(logits.argmax(1).cpu().numpy())
                trues.append(y.cpu().numpy())
    avg = total / max(n, 1)
    if collect_preds:
        return avg, np.concatenate(preds), np.concatenate(trues)
    return avg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Voice Command Model")
    parser.add_argument("--config", required=True, help="Path to a training YAML config")
    parser.add_argument("--exp-id", default=None, help="Override experiment id")
    parser.add_argument("--resume", default=None, help="Resume from a last.pt checkpoint")
    args = parser.parse_args(argv)

    cfg = resolve_config_paths(load_yaml(args.config))
    model_cfg, train_cfg, out_cfg = cfg["model"], cfg["training"], cfg["output"]
    exp_id = args.exp_id or out_cfg.get("exp_id", "EXP")

    set_seed(int(train_cfg.get("seed", 42)))
    device = pick_device(train_cfg.get("device", "auto"))
    print(f"[train] device = {device}")

    labels_path = cfg.get("labels_config", "configs/commands.yaml")
    labels_path = Path(labels_path)
    if not labels_path.is_absolute():
        labels_path = repo_root() / labels_path
    label_map, _ = load_commands_config(labels_path)
    num_classes = label_map.num_classes
    feature_cfg = load_feature_config()
    augment_cfg = load_augment_config()
    # HPC caching uses deterministic base log-Mels. Preserve the default
    # augmentation behavior for every config that does not explicitly opt out.
    augment_cfg.update(train_cfg.get("augmentation", {}) or {})

    feature_kind = model_cfg.get("features", "logmel")
    n_features = int(model_cfg.get("n_features", feature_cfg.n_mels)) if feature_kind == "mfcc" else None
    n_frames = feature_cfg.n_frames

    ds_cfg = cfg["data"]
    cmd_ds = CommandDataset(
        label_map=label_map,
        dataset_configs=ds_cfg.get("datasets", []),
        feature_cfg=feature_cfg,
        augment_cfg=augment_cfg,
        split_cfg=ds_cfg.get("split", {}),
    ).prepare()
    sizes = cmd_ds.sizes()
    if sizes["train"] == 0 or sizes["val"] == 0:
        print("[train] ERROR: empty train or validation split. Aborting.", file=sys.stderr)
        return 1

    model = build_model(model_cfg, num_classes, n_frames, n_features).to(device)
    n_params = count_parameters(model)
    print(f"[train] model = {model_cfg.get('family')}  params = {n_params:,}")

    bs = int(train_cfg.get("batch_size", 64))
    train_set = _TrainSet(
        cmd_ds,
        feature_kind,
        n_features,
        train_cfg.get("feature_augmentation", {}) or {},
    )
    val_set = _EvalSet(cmd_ds, "val", feature_kind, n_features)
    workers = int(train_cfg.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    sampler = None
    if train_cfg.get("sampling") == "balanced":
        # Equalise class exposure per epoch without duplicating files or
        # touching validation/test. Useful for a wake detector where ordinary
        # speech negatives naturally outnumber the wake phrase.
        counts = np.bincount(np.asarray(train_set.labels), minlength=num_classes).astype(float)
        sample_weights = np.asarray([1.0 / max(counts[y], 1.0) for y in train_set.labels])
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_set),
            replacement=True,
        )
        print("[train] balanced sampling enabled")
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, **loader_kwargs)

    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    opt_kwargs = dict(
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    optimizer = (
        torch.optim.Adam(model.parameters(), **opt_kwargs)
        if opt_name == "adam"
        else torch.optim.SGD(model.parameters(), momentum=0.9, **opt_kwargs)
    )

    weight_setting = train_cfg.get("class_weight", "balanced")
    if weight_setting == "balanced":
        class_weights = _class_weights(cmd_ds, num_classes).to(device)
    elif isinstance(weight_setting, list):
        if len(weight_setting) != num_classes:
            raise ValueError("training.class_weight list must match the number of classes")
        class_weights = torch.tensor(weight_setting, dtype=torch.float32, device=device)
    else:
        class_weights = None
    focal_gamma = float(train_cfg.get("focal_gamma", 0.0))
    if focal_gamma > 0:
        class _FocalCrossEntropy(nn.Module):
            def __init__(self, weights, gamma):
                super().__init__(); self.weights = weights; self.gamma = gamma
            def forward(self, logits, labels):
                ce = nn.functional.cross_entropy(logits, labels, weight=self.weights, reduction="none")
                pt = torch.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
                return ((1.0 - pt).pow(self.gamma) * ce).mean()
        criterion = _FocalCrossEntropy(class_weights, focal_gamma)
        print(f"[train] focal loss enabled (gamma={focal_gamma:g})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    sched_name = str(train_cfg.get("lr_scheduler", "cosine")).lower()
    epochs = int(train_cfg.get("epochs", 50))
    scheduler = None
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    ckpt_dir = Path(out_cfg.get("checkpoint_dir", "models/checkpoints")) / exp_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    results_dir = Path(out_cfg.get("results_dir", "results")) / exp_id
    results_dir.mkdir(parents=True, exist_ok=True)

    patience = int(train_cfg.get("early_stopping_patience", 8))
    selection_metric = str(train_cfg.get("selection_metric", "val_macro_f1"))
    if selection_metric not in {"val_macro_f1", "val_loss"}:
        raise ValueError("training.selection_metric must be 'val_macro_f1' or 'val_loss'")
    best_val_f1, bad_epochs, start_epoch = -1.0, 0, 1
    best_selection = float("inf") if selection_metric == "val_loss" else -1.0
    history: list[dict[str, float]] = []

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        if state.get("label_map") != label_map.classes:
            raise ValueError("Resume checkpoint label mapping differs from configs/commands.yaml")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        best_val_f1 = float(state.get("best_val_macro_f1", state.get("val_macro_f1", -1.0)))
        best_selection = float(state.get("best_selection", best_val_f1 if selection_metric == "val_macro_f1" else float("inf")))
        bad_epochs = int(state.get("bad_epochs", 0))
        start_epoch = int(state["epoch"]) + 1
        print(f"[train] resumed {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        tr_loss = _epoch(model, train_loader, criterion, optimizer, device,
                         gradient_clip_norm=train_cfg.get("gradient_clip_norm"))
        val_loss, val_pred, val_true = _epoch(model, val_loader, criterion, None, device, collect_preds=True)
        if scheduler is not None:
            scheduler.step()
        val_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))
        val_acc = float((val_pred == val_true).mean())

        print(
            f"Epoch {epoch}/{epochs}  "
            f"Train Loss: {tr_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  Val MacroF1: {val_f1:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 5),
                "val_macro_f1": round(val_f1, 5),
            }
        )

        selection_value = val_loss if selection_metric == "val_loss" else val_f1
        improved = selection_value < best_selection if selection_metric == "val_loss" else selection_value > best_selection
        if improved:
            best_selection, best_val_f1, bad_epochs = selection_value, val_f1, 0
            checkpoint_state = {
                    "model_state": model.state_dict(),
                    "model_cfg": model_cfg,
                    "num_classes": num_classes,
                    "n_frames": n_frames,
                    "n_features": n_features,
                    "feature_kind": feature_kind,
                    "label_map": label_map.classes,
                    "val_macro_f1": val_f1,
                    "params": n_params,
                    "epoch": epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "best_val_macro_f1": best_val_f1,
                    "selection_metric": selection_metric,
                    "best_selection": best_selection,
                    "bad_epochs": bad_epochs,
                    "seed": int(train_cfg.get("seed", 42)),
                }
            torch.save(checkpoint_state, best_path)
            print(f"  Best model saved -> {best_path.relative_to(repo_root())}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience}).")
                break
        # Persist recoverable state on every completed epoch, including epochs
        # that did not improve validation quality.
        torch.save({
            "model_state": model.state_dict(), "model_cfg": model_cfg,
            "num_classes": num_classes, "n_frames": n_frames, "n_features": n_features,
            "feature_kind": feature_kind, "label_map": label_map.classes, "epoch": epoch,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_val_macro_f1": best_val_f1, "selection_metric": selection_metric,
            "best_selection": best_selection, "bad_epochs": bad_epochs,
            "params": n_params, "seed": int(train_cfg.get("seed", 42)),
        }, last_path)

    with open(results_dir / "training_history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    import yaml

    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (results_dir / "summary.json").write_text(
        json.dumps(
            {
                "exp_id": exp_id,
                "family": model_cfg.get("family"),
                "params": n_params,
                "best_val_macro_f1": round(best_val_f1, 5),
                "selection_metric": selection_metric,
                "best_selection": round(best_selection, 5),
                "epochs_run": len(history),
                "sizes": sizes,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[train] Done. Best val macro-F1 = {best_val_f1:.4f}")
    print(f"[train] Checkpoint: {best_path}")
    print(f"[train] Results:    {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
