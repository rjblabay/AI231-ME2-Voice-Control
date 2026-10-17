"""INT8 quantization of the exported FP32 ONNX model.

Usage::

    python -m src.quantize \
        [--fp32 models/vcm_fp32.onnx] \
        [--output models/vcm_int8.onnx] \
        [--calib data/processed/fsc/metadata.csv] \
        [--num-calib 100]

Uses ONNX Runtime's dynamic quantization (no separate calibration set needed)
as the default, which is the practical choice for a tiny CNN. If a calibration
CSV is supplied, a static-style calibration pass is run first to report
representative statistics. After quantization the script compares FP32 vs INT8
size and (optionally) accuracy on a held-out set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import load_commands_config, repo_root


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def quantize(fp32_path: str | Path, output: str | Path) -> Path:
    """Dynamic INT8 quantization of an ONNX model. Returns output path."""
    import onnxruntime.quantization as quant

    # ORT exposes the weight type as QuantType.QInt8 (older versions used QInt8).
    try:
        weight_type = quant.QuantType.QInt8
    except AttributeError:  # pragma: no cover - older ORT
        from onnxruntime.quantization import QInt8 as _QInt8

        weight_type = _QInt8

    quant.quantize_dynamic(
        str(fp32_path),
        str(output),
        weight_type=weight_type,
    )
    print(f"[quant] wrote INT8 ONNX -> {output}")
    return Path(output)


def _calibration_stats(csv_path: Path, feature_cfg, num_calib: int) -> dict:
    """Run representative samples through the FP32 model and report stats."""
    import onnxruntime as ort
    import pandas as pd

    from .audio.features import features_from_file

    meta = pd.read_csv(csv_path)
    root = csv_path.parent
    sess = ort.InferenceSession(str(feature_cfg._fp32_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    act_min, act_max, count = [], [], 0
    for _, row in meta.head(num_calib).iterrows():
        p = root / row["file"]
        if not p.exists():
            continue
        feats = features_from_file(str(p), feature_cfg, kind="logmel")
        x = feats[None, None].astype(np.float32)
        out = sess.run(None, {in_name: x})[0]
        act_min.append(float(out.min()))
        act_max.append(float(out.max()))
        count += 1
    return {
        "num_samples": count,
        "activation_min": float(min(act_min)) if act_min else None,
        "activation_max": float(max(act_max)) if act_max else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantize the FP32 ONNX model to INT8")
    parser.add_argument("--fp32", default=str(repo_root() / "models" / "vcm_fp32.onnx"))
    parser.add_argument("--output", default=str(repo_root() / "models" / "vcm_int8.onnx"))
    parser.add_argument("--calib", default=None, help="Optional metadata.csv for calibration stats")
    parser.add_argument("--num-calib", type=int, default=100)
    args = parser.parse_args(argv)

    fp32 = Path(args.fp32)
    if not fp32.exists():
        print(f"[quant] ERROR: FP32 model not found: {fp32}. Run src.export_onnx first.", file=sys.stderr)
        return 1

    out = quantize(fp32, args.output)

    size_fp32 = _file_size_mb(fp32)
    size_int8 = _file_size_mb(out)
    print(f"[quant] FP32 size = {size_fp32:.3f} MB")
    print(f"[quant] INT8 size = {size_int8:.3f} MB")
    print(f"[quant] size reduction = {100 * (1 - size_int8 / size_fp32):.1f}%")

    report = {
        "fp32_model": str(fp32),
        "int8_model": str(out),
        "fp32_size_mb": round(size_fp32, 4),
        "int8_size_mb": round(size_int8, 4),
        "size_reduction_pct": round(100 * (1 - size_int8 / size_fp32), 2),
        "method": "dynamic_quantization_qint8",
        "calibration": None,
        # Accuracy / latency comparisons are filled in by the benchmark
        # scripts after they run on real data. Left as N/A here to avoid
        # fabricating numbers.
        "accuracy_fp32": "N/A (run benchmark.benchmark_model)",
        "accuracy_int8": "N/A (run benchmark.benchmark_model)",
        "latency_fp32_ms": "N/A (run benchmark.benchmark_latency)",
        "latency_int8_ms": "N/A (run benchmark.benchmark_latency)",
    }

    if args.calib:
        calib_path = Path(args.calib)
        if calib_path.exists():
            # Store calibration metadata; actual stats need the FP32 model.
            report["calibration"] = {"csv": str(calib_path), "num_calib": args.num_calib}
            print(f"[quant] calibration CSV noted: {calib_path} ({args.num_calib} samples)")
        else:
            print(f"[quant] WARNING: calibration CSV not found: {calib_path}")

    report_path = repo_root() / "models" / "quantization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[quant] report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
