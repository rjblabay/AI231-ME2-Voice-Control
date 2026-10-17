#!/usr/bin/env python3
"""Choose a conservative Astra threshold on validation, then score test once."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.features import load_feature_config
from src.config import LabelMap, load_yaml, resolve_config_paths
from src.data.datasets import CommandDataset
from src.models.factory import build_model
from src.train import _EvalSet


def scores(model, ds, split, device):
    loader = DataLoader(_EvalSet(ds, split, "logmel", None), batch_size=64, shuffle=False)
    ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.extend(torch.softmax(model(x.to(device)), 1)[:, 0].cpu().numpy())
            ys.extend(y.numpy())
    return np.asarray(ps), np.asarray(ys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    a = p.parse_args()
    cfg = resolve_config_paths(load_yaml(a.config)); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location=device)
    model = build_model(ckpt["model_cfg"], ckpt["num_classes"], ckpt["n_frames"], ckpt["n_features"]).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    labels = LabelMap(ckpt["label_map"])
    ds = CommandDataset(labels, cfg["data"]["datasets"], load_feature_config(), {"enabled": False}, cfg["data"]["split"]).prepare()
    val_p, val_y = scores(model, ds, "val", device)
    # On equal macro-F1 pick the highest threshold: fewer avoidable false wakes.
    candidates = np.linspace(0.50, 0.99, 99)
    threshold = max(candidates, key=lambda t: (
        f1_score(val_y, np.where(val_p >= t, 0, 1), average="macro", zero_division=0), t
    ))
    test_p, test_y = scores(model, ds, "test", device); pred = (test_p >= threshold).astype(int)
    # label 0 is ASTRA, so invert the boolean prediction.
    pred = np.where(pred == 1, 0, 1)
    metrics = {"threshold": round(float(threshold), 3), "accuracy": round(float(accuracy_score(test_y, pred)), 4),
               "macro_f1": round(float(f1_score(test_y, pred, average="macro", zero_division=0)), 4),
               "astra_precision": round(float(precision_score(test_y, pred, labels=[0], average="macro", zero_division=0)), 4),
               "astra_recall": round(float(recall_score(test_y, pred, labels=[0], average="macro", zero_division=0)), 4),
               "val_macro_f1": round(float(f1_score(val_y, np.where(val_p >= threshold, 0, 1), average="macro", zero_division=0)), 4)}
    Path(a.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
