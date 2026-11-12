# SAMII-TCR

Meta-learning framework for TCR–pMHC binding prediction with few-shot
test-time adaptation. SAMII-TCR couples a supervised attention baseline with a
Reptile-style meta-learner that adapts to a new task (epitope–MHC pair) from a
small support set via test-time fine-tuning (TTFT) with L2 anchoring.

## Repository layout

```
SAMII-TCR/
├── data/
│   ├── 10x/ 100x/ 1000x/        # benchmark splits (meta_train, meta_test,
│   │                            #   few_test_support, few_test_query, unseen_test)
│   ├── split_tasks.py           # task-disjoint split of the positive dataset
│   ├── generate_negatives.py    # random negative backgrounds (pmhc_not_bind + healthy TCR)
│   └── generate_matched_background.py  # chain-status & length matched healthy backgrounds
├── models/
│   ├── dataset.py               # featurization (CDR3 / epitope / MHC pseudo-sequence)
│   ├── model.py                 # attention model + metrics
│   ├── train_baseline.py        # supervised baseline, 5-fold task-grouped CV
│   ├── train_meta.py            # meta-training (Reptile outer loop + TTFT)
│   ├── predict.py               # ensemble prediction with task routing
│   └── predict_baseline.py      # ensemble prediction with the baseline
├── mhc_pseudoseq/               # MHC class II pseudo-sequences
├── trained_models/
│   ├── chain_matched/           # models trained on chain-matched backgrounds (primary)
│   └── random_background/       # models trained on randomly sampled healthy backgrounds
└── evaluation/
    ├── run_evaluation.sh        # reproduce the test-set evaluation
    ├── summarize_results.py     # metrics table + bar chart
    └── expected_results.csv     # reference metrics
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

Each ratio directory (`10x`, `100x`, `1000x`) contains the same positive
records with different negative backgrounds. Negatives per positive: 5x
`pmhc_not_bind` (TCR kept, pMHC replaced) plus Nx healthy-donor TCRs, where N =
10/100/1000. In the primary `chain_matched` setting the healthy TCRs are drawn
from the positive's own chain-status layer (paired / TRA-only / TRB-only) with
CDR3 total-length matching (|Δ| ≤ 4 → ≤ 8 → any within layer, with
replacement). Task identity is `epitope|MHCA_norm|MHCB_norm`; partitions are
task-disjoint: seen tasks (>10 positives) → meta_train/meta_test, few-shot
tasks (5–10) → support/query, unseen tasks (<5) → unseen_test.

To regenerate the splits from raw data, place the combined positive dataset and
the healthy TCR pool under `data/raw/` and run:

```bash
cd data
python split_tasks.py
python generate_negatives.py            # random background
M_MATCH_MULT=10  M_TAG=10x   python generate_matched_background.py
M_MATCH_MULT=100 M_TAG=100x  python generate_matched_background.py
M_MATCH_MULT=1000 M_TAG=1000x python generate_matched_background.py
```

## Reproduce the benchmark results

With the released weights (chain_matched, seed 2025), run:

```bash
bash evaluation/run_evaluation.sh            # 1000x
TIER=100x bash evaluation/run_evaluation.sh
TIER=10x  bash evaluation/run_evaluation.sh
```

This writes per-split prediction CSVs to `evaluation/predictions/` and prints a
metric table together with the difference from `expected_results.csv`:

| Split | SAMII-TCR ROC-AUC / PR-AUC | Baseline ROC-AUC / PR-AUC |
|---|---|---|
| 1000x meta_test | 0.8792 / 0.2846 | 0.8622 / 0.2654 |
| 1000x few_test_query | 0.9468 / 0.0328 | 0.9296 / 0.0106 |
| 1000x unseen_test | 0.9339 / 0.0284 | 0.9355 / 0.0185 |

(10x and 100x reference values are in `evaluation/expected_results.csv`.)

The two model sets under `trained_models/` differ only in the negative
background used for training: `chain_matched` (primary) uses healthy
backgrounds matched to the positives by chain status and CDR3 length, whereas
`random_background` uses randomly sampled healthy TCRs. To evaluate the
random-background weights instead:

```bash
MODEL_SET=random_background bash evaluation/run_evaluation.sh
```

## Prediction on new data

```bash
python models/predict.py \
    --meta_train data/1000x/meta_train.csv.gz \
    --test_file your_queries.csv \
    --support_file your_support.csv   # optional; enables few-shot TTFT
    --output prediction_result.csv
```

Input CSV columns: `cdr3_TRA`, `cdr3_TRB` (either may be empty),
`epitope`, `MHCA`, `MHCB`, and optionally `Target` (1/0) for evaluation.
Records whose task appears in the training set are routed to the baseline
ensemble; unseen tasks with a support set are predicted after TTFT; unseen
tasks without support are predicted zero-shot.

## Training from scratch

```bash
# baseline (all ratios, or RUN_VARIANTS=1000x for one)
cd models
python train_baseline.py

# meta model (per ratio)
NEG_RATIO=1000x python train_meta.py
```

Set `TRAIN_SEED` to change the random seed (default 2025) and `MODEL_SET` to
choose the output collection (`chain_matched` or `random_background`).
