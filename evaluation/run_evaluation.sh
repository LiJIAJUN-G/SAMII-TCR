#!/bin/bash
# Reproduce the test-set evaluation of SAMII-TCR and the supervised baseline.
#
# Usage (run from anywhere):
#   bash evaluation/run_evaluation.sh              # 1000x, chain_matched models
#   TIER=100x bash evaluation/run_evaluation.sh    # other ratio
#   MODEL_SET=random_background bash evaluation/run_evaluation.sh
set -e
cd "$(dirname "$0")/.."   # repository root

TIER=${TIER:-1000x}
MODEL_SET=${MODEL_SET:-chain_matched}

DATA=data/${TIER}
META_DIR=trained_models/${MODEL_SET}/meta_${TIER}
BASE_DIR=trained_models/${MODEL_SET}/baseline_${TIER}
TRAIN=${DATA}/meta_train.csv.gz
[ -f "$TRAIN" ] || TRAIN=${DATA}/meta_train.csv
META_TEST=${DATA}/meta_test.csv.gz
[ -f "$META_TEST" ] || META_TEST=${DATA}/meta_test.csv

OUT=evaluation/predictions/${MODEL_SET}/${TIER}
mkdir -p "$OUT"

echo "=== Evaluating tier=${TIER} model_set=${MODEL_SET} ==="

# ---- SAMII-TCR (meta model) ----
# meta_test: evaluate the meta model itself on the seen-task partition
python models/predict.py --force_meta \
    --meta_train "$TRAIN" \
    --test_file "$META_TEST" \
    --meta_dir "$META_DIR" --baseline_dir "$BASE_DIR" \
    --output "$OUT/meta_test_pred_samiitcr.csv"

# unseen_test: the meta model is evaluated on the whole partition
# (--force_meta: some pmhc_not_bind negatives carry rebuilt task_ids that also
#  appear among meta_train negatives; the SAMII-TCR column reports the meta
#  model itself, so all unseen_test records are routed to it)
python models/predict.py --force_meta \
    --meta_train "$TRAIN" \
    --test_file "${DATA}/unseen_test.csv" \
    --meta_dir "$META_DIR" --baseline_dir "$BASE_DIR" \
    --output "$OUT/unseen_test_pred_samiitcr.csv"

# few_test_query: TTFT on the support set
python models/predict.py \
    --meta_train "$TRAIN" \
    --support_file "${DATA}/few_test_support.csv" \
    --test_file "${DATA}/few_test_query.csv" \
    --meta_dir "$META_DIR" --baseline_dir "$BASE_DIR" \
    --output "$OUT/few_test_query_pred_samiitcr.csv"

# ---- Supervised baseline ----
for SPLIT in meta_test unseen_test few_test_query; do
    IN=${DATA}/${SPLIT}.csv.gz
    [ -f "$IN" ] || IN=${DATA}/${SPLIT}.csv
    python models/predict_baseline.py \
        --input_csv "$IN" \
        --model_dir "$BASE_DIR" \
        --output_csv "$OUT/${SPLIT}_pred_baseline.csv"
done

# ---- Summarize ----
python evaluation/summarize_results.py \
    --pred_dir "$OUT" \
    --tier "$TIER" \
    --out_prefix "evaluation/Comparison_Eval_${MODEL_SET}_${TIER}"

echo "=== Done. Predictions in $OUT ==="
