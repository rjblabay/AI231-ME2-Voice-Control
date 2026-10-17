"""Export a trained PyTorch checkpoint to ONNX and verify agreement.

Usage::

    python -m src.export_onnx --checkpoint models/checkpoints/best_dscnn.pt \
        [--output models/vcm_fp32.onnx]

After export, several random inputs are run through both PyTorch and ONNX
Runtime and the outputs are compared. The command fails loudly if they differ
beyond a small tolerance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .config import load_commands_config, repo_root
from .models.factory import build_model


def export(
    checkpoint: str | Path,
    output: str | Path,
    tolerance: float = 1e-3,
) -> Path:
    """Export to ONNX and verify PyTorch vs ONNX agreement. Returns output path."""
    import onnxruntime as ort

    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(
        ckpt["model_cfg"], ckpt["num_classes"], ckpt["n_frames"], ckpt["n_features"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_frames = ckpt["n_frames"]
    # For log-mel models the checkpoint stores n_features=None; the feature
    # width is n_mels. Derive it so the dummy input has a concrete shape.
    n_features = ckpt["n_features"]
    if n_features is None:
        n_features = int(ckpt["model_cfg"].get("n_mels", 40))
    # Input is (1, T, F) for MLP and CRNN-BiGRU; CNN/DS-CNN accept 4-D.
    family = ckpt["model_cfg"].get("family")
    if family in {"mlp", "crnn_bigru", "crnn_bigru_verifier"}:
        in_shape = (1, n_frames, n_features)
    else:
        in_shape = (1, 1, n_frames, n_features)

    dummy = torch.randn(*in_shape)
    opset = 13
    # Use the legacy TorchScript exporter - it is more permissive than the
    # newer dynamo-based exporter and produces stable, small ONNX graphs for
    # these simple CNN/MLP models.
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] wrote ONNX -> {output}")

    # ---- verify agreement ----
    sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(8):
        x_np = rng.standard_normal(in_shape).astype(np.float32)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(x_np)).numpy()
        onnx_out = sess.run(None, {in_name: x_np})[0]
        diff = float(np.max(np.abs(pt_out - onnx_out)))
        max_diff = max(max_diff, diff)

    if max_diff > tolerance:
        print(
            f"[export] ERROR: PyTorch vs ONNX outputs differ by {max_diff:.2e} "
            f"(tolerance {tolerance:.1e}). Export FAILED.",
            file=sys.stderr,
        )
        raise RuntimeError("ONNX export verification failed")
    print(f"[export] PyTorch vs ONNX max abs diff = {max_diff:.2e} (OK, tol {tolerance:.1e})")
    return Path(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=str(repo_root() / "models" / "vcm_fp32.onnx"))
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args(argv)

    export(args.checkpoint, args.output, args.tolerance)
    print(f"[export] Done. Model at: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
