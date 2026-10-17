# ME2 Voice Control — ASTRA

ASTRA is an on-device Raspberry Pi voice assistant with an **Astra** wake-word
verifier and a CRNN-BiGRU voice-command model. Inference uses ONNX Runtime on
the Raspberry Pi 4 CPU; no cloud ASR or browser-side inference is required.

## Released artifacts

This repository commits the final trained checkpoints, ONNX exports, metrics,
learning curves, model configurations, and HPC logs:

| Artifact | Model | Test result |
| --- | --- | --- |
| `models/macbook/ASTRA_CRNN_BIGRU_VERIFIER_20261016_184000/` | Astra wake verifier | 98.44% accuracy, 82.93% macro-F1 |
| `models/macbook/AI231_HPO_NOSAMPLER_20261016_210738/` | 99-class combined phrase/slot command classifier | 90.16% accuracy, 89.20% macro-F1 |

The command classifier has a single 99-class phrase-and-slot output: 93
in-vocabulary command variations, five unsupported-slot rejection classes,
and `OUT_OF_SCOPE`. Runtime normalizes these to 19 semantic commands.

## Quick start

Install development dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the deployed Pi listener through the existing Mac launcher:

```bash
./run_astra_headless_mac.sh
```

For a one-command command-model reproduction (downloads the public source
data, prepares it, trains, evaluates, and exports ONNX):

```bash
./scripts/reproduce_command_model.sh --download
```

The script intentionally does not train on the dataset's live `holdout` split.

## Data, evaluation, and responsible release

The command corpus is the citable Hugging Face dataset
[`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands),
DOI [`10.57967/hf/10723`](https://doi.org/10.57967/hf/10723).

Raw audio is **not** committed. It includes mixed source licences and
class-owned recordings; obtain it from Hugging Face and follow the source
terms. Code is MIT-licensed. The trained weights are research/education-only,
non-commercial and inherit dataset restrictions.

See:

- [Evaluation protocol and held-out unseen-speaker test](docs/evaluation_protocol.md)
- [Raspberry Pi 4 latency reproduction](docs/pi4_benchmark.md)
- [`configs/assistant.env.example`](configs/assistant.env.example) for local weather credentials

## Pi 4 performance

On a Raspberry Pi 4 Model B, 19 live trials measured p95 ONNX inference of
**116.5 ms (0.039× RTF)** for wake-word detection and **176.0 ms (0.059×
RTF)** for command classification. The end-to-end interaction p95 was 3.994 s
and includes user speech capture plus the configured command delay.

## Training provenance

The final command model was trained on the UP HPC A100 cluster with one
NVIDIA A100-SXM4-40GB. Training seed: `20261019`; data-split seed:
`20261016`. Final HPO logs are in `logs/hpc/`.
