"""Evaluation CLI - full metrics, confusion matrix, and plots.

Usage::

    python -m src.evaluate \
        --checkpoint models/checkpoints/best_dscnn.pt \
        --config configs/dscnn.yaml \
        --split test

Writes to ``results/<exp_id>/``:
    metrics.json, per_class.csv, confusion_matrix.png, and (if a training
    history exists) training_loss.png / training_f1.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .audio.features import load_feature_config
from .config import LabelMap, load_commands_config, load_yaml, pick_device, repo_root, resolve_config_paths, set_seed
from .data.datasets import CommandDataset
from .models.factory import build_model
from .train import _EvalSet


def _plot_confusion(cm, labels, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.set_title("Confusion Matrix (rows=true, cols=pred)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_history(hist: list[dict], path_loss: Path, path_f1: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in hist]
    tr = [h["train_loss"] for h in hist]
    vl = [h["val_loss"] for h in hist]
    vf = [h["val_macro_f1"] for h in hist]

    fig, ax = plt.subplots()
    ax.plot(epochs, tr, label="train loss")
    ax.plot(epochs, vl, label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("Training Loss")
    fig.tight_layout()
    fig.savefig(path_loss, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, vf, marker="o", label="val macro-F1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("macro-F1")
    ax.legend()
    ax.set_title("Validation Macro-F1")
    fig.tight_layout()
    fig.savefig(path_f1, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Voice Command Model")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    parser.add_argument("--config", required=True, help="Training YAML config (for data + model shape)")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--exp-id", default=None)
    args = parser.parse_args(argv)

    cfg = resolve_config_paths(load_yaml(args.config))
    model_cfg, out_cfg = cfg["model"], cfg["output"]
    exp_id = args.exp_id or out_cfg.get("exp_id", "EXP")
    set_seed(0)
    device = pick_device("auto")

    feature_cfg = load_feature_config()

    ckpt = torch.load(args.checkpoint, map_location=device)
    # A checkpoint is self-describing. This keeps the preserved 20-class
    # baseline evaluable after taxonomy extensions and supports a separate wake model.
    label_map = LabelMap(ckpt["label_map"])
    num_classes = label_map.num_classes
    n_frames = ckpt.get("n_frames", feature_cfg.n_frames)
    n_features = ckpt.get("n_features", feature_cfg.n_mels)
    feature_kind = ckpt.get("feature_kind", "logmel")
    model = build_model(ckpt["model_cfg"], ckpt["num_classes"], n_frames, n_features).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] loaded {args.checkpoint} (family={ckpt['model_cfg'].get('family')}, params={ckpt.get('params'):,})")

    ds_cfg = cfg["data"]
    cmd_ds = CommandDataset(
        label_map=label_map,
        dataset_configs=ds_cfg.get("datasets", []),
        feature_cfg=feature_cfg,
        augment_cfg={"enabled": False},  # never augment at eval
        split_cfg=ds_cfg.get("split", {}),
    ).prepare()

    eval_set = _EvalSet(cmd_ds, args.split, feature_kind, n_features)
    loader = DataLoader(eval_set, batch_size=64, shuffle=False)

    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    acc = float((preds == trues).mean())
    macro_p = float(precision_score(trues, preds, average="macro", zero_division=0))
    macro_r = float(recall_score(trues, preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(trues, preds, average="macro", zero_division=0))

    per_class = {}
    for ci, cname in enumerate(label_map.classes):
        p = float(precision_score(trues, preds, labels=[ci], average="macro", zero_division=0))
        r = float(recall_score(trues, preds, labels=[ci], average="macro", zero_division=0))
        f = float(f1_score(trues, preds, labels=[ci], average="macro", zero_division=0))
        per_class[cname] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    cm = confusion_matrix(trues, preds, labels=list(range(num_classes)))

    results_dir = Path(out_cfg.get("results_dir", "results")) / exp_id
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "exp_id": exp_id,
        "split": args.split,
        "num_samples": int(len(eval_set)),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }
    (results_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # per-class CSV
    import csv

    with open(results_dir / f"per_class_{args.split}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "precision", "recall", "f1"])
        for cname, m in per_class.items():
            w.writerow([cname, m["precision"], m["recall"], m["f1"]])

    # confusion matrix plot
    _plot_confusion(cm, label_map.classes, results_dir / f"confusion_matrix_{args.split}.png")

    # training history plots if present
    hist_path = results_dir / "training_history.csv"
    if hist_path.exists():
        import pandas as pd

        hist = pd.read_csv(hist_path).to_dict("records")
        _plot_history(hist, results_dir / "training_loss.png", results_dir / "training_f1.png")

    print(f"\n[eval] {args.split} set: n={len(eval_set)}")
    print(f"[eval] Accuracy={acc:.4f}  MacroP={macro_p:.4f}  MacroR={macro_r:.4f}  MacroF1={macro_f1:.4f}")
    print(f"[eval] Outputs -> {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
