# Evaluation protocol

## Data provenance and separation

The command model is trained from the public AI231 ME2 Voice Commands dataset:

- Dataset: [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)
- DOI: [`10.57967/hf/10723`](https://doi.org/10.57967/hf/10723)

Raw dataset audio is deliberately not mirrored in this repository. The dataset
combines sources with their own licences, including class-owned recordings.
Users must obtain it from the dataset publisher and comply with its
research/education-only, non-commercial terms and each source's terms.

The published command-model training run used a speaker-disjoint partition:

| Partition | Utterances | Speakers | Role |
| --- | ---: | ---: | --- |
| Train | 10,071 | 558 | Training |
| Validation | 1,662 | 99 | Model selection / HPO |
| Test | 4,693 | 214 | Untouched offline evaluation |

The public dataset also has a separate live Raspberry Pi `holdout` split. It
is not used by `scripts/reproduce_command_model.sh` and is not copied here.

## Final command model

`AI231_HPO_NOSAMPLER_20261016_210738` was selected by validation macro-F1.
Its saved test report is committed beside the checkpoint in
`models/macbook/AI231_HPO_NOSAMPLER_20261016_210738/results/`.

- Test accuracy: **90.16%**
- Test macro-F1: **89.20%**
- Parameters: **1,349,404**

## Comparable baseline

The comparable CRNN-BiGRU baseline (`AI231_CRNN_BIGRU_20261016_200000`) used
the same prepared data partition and a smaller **897,796-parameter** model.

| Model | Parameters | Test accuracy | Test macro-F1 |
| --- | ---: | ---: | ---: |
| CRNN-BiGRU baseline | 897,796 | 84.70% | 83.41% |
| Final CRNN-BiGRU HPO | 1,349,404 | 90.16% | 89.20% |

The final model is 1.50× the parameter count of the baseline; this is a
comparable-size architectural/HPO comparison, not a comparison with a large
cloud model.
