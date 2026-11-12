# -*- coding: utf-8 -*-
"""
Summarize prediction CSVs into a metric table (ROC-AUC / PR-AUC) for
SAMII-TCR and the supervised baseline on meta_test / few_test_query /
unseen_test, compare against the expected reference values, and draw a
two-panel bar chart (Comparison_Eval style).

Usage:
    python evaluation/summarize_results.py --pred_dir evaluation/predictions/chain_matched/1000x \
        --tier 1000x --out_prefix evaluation/Comparison_Eval_chain_matched_1000x
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 13

COLOR_SAMII = "#cb4e48"
COLOR_BASE = "#f2cf8d"

SPLITS = ["meta_test", "few_test_query", "unseen_test"]
SPLIT_LABELS = {"meta_test": "Meta Test", "few_test_query": "Few-Shot Query", "unseen_test": "Unseen Test"}


def metrics_from_csv(path, prob_col):
    df = pd.read_csv(path)
    df = df.dropna(subset=[prob_col])
    y = df["label"].astype(float).values
    p = df[prob_col].astype(float).values
    return roc_auc_score(y, p), average_precision_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--tier", default="1000x")
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    rows = []
    for split in SPLITS:
        p_samii = os.path.join(args.pred_dir, f"{split}_pred_samiitcr.csv")
        p_base = os.path.join(args.pred_dir, f"{split}_pred_baseline.csv")
        if os.path.exists(p_samii):
            r, pr = metrics_from_csv(p_samii, "hybrid_prob")
            rows.append({"dataset": f"{args.tier}_{split}", "model": "SAMII-TCR",
                         "ROC_AUC": round(r, 4), "PR_AUC": round(pr, 4)})
        if os.path.exists(p_base):
            r, pr = metrics_from_csv(p_base, "pred_prob")
            rows.append({"dataset": f"{args.tier}_{split}", "model": "Baseline",
                         "ROC_AUC": round(r, 4), "PR_AUC": round(pr, 4)})

    df = pd.DataFrame(rows)
    out_csv = (args.out_prefix or os.path.join(args.pred_dir, "summary")) + ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")

    # compare with expected reference values when available
    exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_results.csv")
    if os.path.exists(exp_path) and "chain_matched" in args.pred_dir:
        exp = pd.read_csv(exp_path)
        exp = exp[exp["dataset"].str.startswith(f"{args.tier}_")]
        m = df.merge(exp, on=["dataset", "model"], suffixes=("", "_expected"))
        m["dROC"] = (m["ROC_AUC"] - m["ROC_AUC_expected"]).round(4)
        m["dPR"] = (m["PR_AUC"] - m["PR_AUC_expected"]).round(4)
        print("\nComparison with expected reference values:")
        print(m[["dataset", "model", "ROC_AUC", "ROC_AUC_expected", "dROC",
                 "PR_AUC", "PR_AUC_expected", "dPR"]].to_string(index=False))

    if len(df) == 0:
        return

    # bar chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, ylabel in zip(axes, ["ROC_AUC", "PR_AUC"], ["ROC-AUC", "PR-AUC"]):
        x = np.arange(len(SPLITS))
        w = 0.36
        vals_s, vals_b = [], []
        for s in SPLITS:
            vs = df[(df["dataset"] == f"{args.tier}_{s}") & (df["model"] == "SAMII-TCR")][metric]
            vb = df[(df["dataset"] == f"{args.tier}_{s}") & (df["model"] == "Baseline")][metric]
            vals_s.append(vs.iloc[0] if len(vs) else 0)
            vals_b.append(vb.iloc[0] if len(vb) else 0)
        b1 = ax.bar(x - w / 2, vals_s, w, color=COLOR_SAMII, label="SAMII-TCR")
        b2 = ax.bar(x + w / 2, vals_b, w, color=COLOR_BASE, label="Baseline")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(vals_s), max(vals_b)) * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "ROC_AUC":
            ax.legend(frameon=False, loc="lower left")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{os.path.splitext(out_csv)[0]}.{ext}", dpi=300)
    print(f"Saved figure: {os.path.splitext(out_csv)[0]}.pdf/.png")


if __name__ == "__main__":
    main()
