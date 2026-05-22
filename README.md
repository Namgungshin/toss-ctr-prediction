# Toss Ads CTR Prediction

DACON Toss ads CTR prediction competition workspace.

This repository contains the reproducible experiment code and notes used for the
first baseline cycle. Competition raw data and generated submissions are not
tracked because they are large and should be downloaded from DACON directly.

## Competition

- Task: binary click-through-rate prediction for ad impressions
- Target: `clicked`
- Evaluation: AP and weighted log loss
- Data shape observed locally:
  - train: 10,704,179 rows, 119 columns
  - test: 1,527,298 rows, 119 columns
- Important split signal: all test rows have `day_of_week = 7`

## Repository Contents

| path | purpose |
|---|---|
| `metrics.py` | Local AP, weighted log loss, and proxy metric helpers |
| `eda_summary.py` | Generates compact parquet/schema/distribution summary |
| `train_baseline.py` | LightGBM baseline with day7 or random validation |
| `make_submission.py` | Creates a DACON submission from a saved model |
| `calibrate_submission.py` | Prior-calibration experiment, kept for record |
| `blend_submissions.py` | Simple submission blending utility |
| `reports/eda_summary.md` | EDA snapshot generated from local data |
| `reports/experiment_log.md` | Submission and experiment score log |

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Place the DACON files in the project root:

```text
train.parquet
test.parquet
sample_submission.csv
```

## Reproduce Current Baseline

Generate EDA summary:

```bash
python3 eda_summary.py --output reports/eda_summary.md
```

Train the first day7-validation baseline:

```bash
python3 train_baseline.py \
  --output-dir models/baseline_lgbm \
  --train-sample-frac 0.10 \
  --valid-sample-frac 0.30 \
  --seed 42
```

Create the submission:

```bash
python3 make_submission.py \
  --model-dir models/baseline_lgbm \
  --output submissions/baseline_lgbm.csv
```

Train with basic `seq` features:

```bash
python3 train_baseline.py \
  --output-dir models/baseline_lgbm_seq \
  --train-sample-frac 0.10 \
  --valid-sample-frac 0.30 \
  --use-seq-features \
  --seed 42
```

## Current Results

| submission | local validation | public score | note |
|---|---:|---:|---|
| `baseline_lgbm.csv` | AP 0.05833 / WLL 0.61761 | 0.340445203 | First raw LightGBM baseline |
| `baseline_lgbm_prior_calibrated.csv` | n/a | 0.2045772218 | Prior calibration hurt badly |
| `baseline_lgbm_seq.csv` | AP 0.05890 / WLL 0.61836 | pending | Basic sequence features |

## Takeaways

- Use `day_of_week = 7` validation as the main local validation because the test
  set is day 7 only.
- Do not calibrate predictions down to the real train CTR prior; the public
  score got much worse.
- Next high-value experiments are count encoding, target encoding with leakage
  controls, and better sequence-derived features.
