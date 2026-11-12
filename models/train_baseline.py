# -*- coding: utf-8 -*-
"""
Train the supervised baseline model (attention encoder without the meta-learned
9-mer register head) with 5-fold cross-validation grouped by task_id.

For each negative-to-positive ratio (10x / 100x / 1000x), the meta_train pool is
split into 5 task-disjoint folds; one model is trained per fold and evaluated on
all held-out test partitions (meta_test, few_test_query, unseen_test).

Outputs are written to trained_models/<model_set>/baseline_<ratio>/fold_<k>/.

Usage:
    python train_baseline.py                     # all ratios, chain_matched set
    RUN_VARIANTS=1000x python train_baseline.py  # single ratio
    MODEL_SET=random_background python train_baseline.py
"""

import os
import json
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PMHCTCRDataset, preprocess
from model import AttBaselineModel, set_seed, evaluate


CFG = {
    "seed": int(os.environ.get("TRAIN_SEED", "2025")),
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # data/<ratio>/{meta_train,meta_test,few_test_support,few_test_query,unseen_test}.csv
    "data_root": os.environ.get("DATA_ROOT", "data"),

    # negative-to-positive ratios to train on
    "variants": ["10x", "100x", "1000x"],

    "train_file": "meta_train.csv.gz",

    "test_files": {
        "meta_test": "meta_test.csv.gz",
        "few_test_query": "few_test_query.csv",
        "unseen_test": "unseen_test.csv",
    },

    "also_eval_merged_test": True,

    # group column used to keep tasks disjoint across folds
    "task_col": "task_id",

    "n_folds": 5,

    "batch_size": 256,
    "num_workers": 4,
    "pin_memory": True,

    "d_model": 256,
    "nhead": 8,
    "num_layers": 1,
    "ffn_dim": 512,
    "dropout": 0.1,
    "max_cdr3": 60,
    "max_pep": 25,
    "max_mhc": 128,

    "epochs": 30,
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,

    # early stopping on fold-validation metric
    "monitor": "PR_AUC",
    "patience": 5,
    "min_delta": 1e-4,

    "out_root": os.environ.get(
        "OUT_ROOT",
        os.path.join("trained_models", os.environ.get("MODEL_SET", "chain_matched")),
    ),
}

_env_variants = os.environ.get("RUN_VARIANTS", "").strip()
RUN_VARIANTS = (_env_variants.split(",") if _env_variants
                else ["10x", "100x", "1000x"])


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_csv(path: str) -> pd.DataFrame:
    # accept either plain .csv or gzipped .csv.gz
    if not os.path.exists(path):
        alt = path[:-3] if path.endswith(".gz") else path + ".gz"
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(path)
    return pd.read_csv(path)

def build_loader(df: pd.DataFrame, batch_size: int, shuffle: bool,
                 num_workers: int, pin_memory: bool,
                 max_cdr3: int, max_pep: int, max_mhc: int) -> DataLoader:
    ds = PMHCTCRDataset(df, max_cdr3=max_cdr3, max_pep=max_pep, max_mhc=max_mhc)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

def train_one_epoch(model, loader, optimizer, device, loss_fn, grad_clip: float = 0.0):
    model.train()
    total_loss = 0.0
    n = 0
    for b in loader:
        for k in ["cdr3", "pep", "mhc", "y"]:
            b[k] = b[k].to(device)

        optimizer.zero_grad(set_to_none=True)
        logit, _ = model(b)
        loss = loss_fn(logit, b["y"])
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = b["y"].shape[0]
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(1, n)

def eval_one_split(model, df_raw: pd.DataFrame, device, split_name: str, out_dir: str):
    df_p = preprocess(df_raw)
    loader = build_loader(
        df_p,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=CFG["pin_memory"],
        max_cdr3=CFG["max_cdr3"],
        max_pep=CFG["max_pep"],
        max_mhc=CFG["max_mhc"],
    )
    metrics = evaluate(model, loader, device, return_arrays=False)
    save_json(metrics, os.path.join(out_dir, f"test_metrics_{split_name}.json"))
    return metrics

def evaluate_on_variant(model, eval_variant: str, device, out_dir: str):
    """Evaluate one trained fold model on every test partition of one ratio."""
    data_dir = os.path.join(CFG["data_root"], eval_variant)

    results = {}
    for split_name, file_name in CFG["test_files"].items():
        p = os.path.join(data_dir, file_name)
        df_raw = load_csv(p)
        results[split_name] = eval_one_split(model, df_raw, device, split_name, out_dir)

    if CFG["also_eval_merged_test"]:
        dfs = [load_csv(os.path.join(data_dir, f)) for f in CFG["test_files"].values()]
        merged_raw = pd.concat(dfs, axis=0, ignore_index=True)
        results["merged_all_tests"] = eval_one_split(model, merged_raw, device, "merged_all_tests", out_dir)

    save_json(results, os.path.join(out_dir, "all_test_results.json"))
    return results

def make_group_folds(df: pd.DataFrame, group_col: str, n_splits: int, seed: int):
    """
    Group-aware K-fold split: rows sharing the same group (task_id) never appear
    in both the training and validation part of a fold. Returns a list of
    (train_idx, val_idx) row-index pairs.
    """
    if group_col not in df.columns:
        raise ValueError(f"missing group column {group_col}")

    groups = df[group_col].astype(str).values
    uniq = np.unique(groups)

    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    fold_tasks = [set() for _ in range(n_splits)]
    for i, t in enumerate(uniq):
        fold_tasks[i % n_splits].add(t)

    folds = []
    idx_all = np.arange(len(df))
    for k in range(n_splits):
        val_mask = np.isin(groups, list(fold_tasks[k]))
        val_idx = idx_all[val_mask]
        train_idx = idx_all[~val_mask]
        folds.append((train_idx, val_idx))
    return folds


def run_train_variant_5fold(train_variant: str):
    set_seed(CFG["seed"])
    device = torch.device(CFG["device"])

    train_data_dir = os.path.join(CFG["data_root"], train_variant)
    out_base = os.path.join(CFG["out_root"], f"baseline_{train_variant}")
    _ensure_dir(out_base)

    df_train_raw_all = load_csv(os.path.join(train_data_dir, CFG["train_file"]))
    df_train_all = preprocess(df_train_raw_all)

    folds = make_group_folds(df_train_all, CFG["task_col"], CFG["n_folds"], CFG["seed"])

    fold_results = {}

    for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_dir = os.path.join(out_base, f"fold_{fold_id}")
        _ensure_dir(fold_dir)

        df_tr = df_train_all.iloc[tr_idx].reset_index(drop=True)
        df_va = df_train_all.iloc[va_idx].reset_index(drop=True)

        tr_loader = build_loader(
            df_tr, CFG["batch_size"], True,
            CFG["num_workers"], CFG["pin_memory"],
            CFG["max_cdr3"], CFG["max_pep"], CFG["max_mhc"]
        )
        va_loader = build_loader(
            df_va, CFG["batch_size"], False,
            CFG["num_workers"], CFG["pin_memory"],
            CFG["max_cdr3"], CFG["max_pep"], CFG["max_mhc"]
        )

        model = AttBaselineModel(
            d_model=CFG["d_model"],
            nhead=CFG["nhead"],
            num_layers=CFG["num_layers"],
            ffn_dim=CFG["ffn_dim"],
            attn_dropout=CFG["dropout"],
            emb_dropout=CFG["dropout"],
            max_cdr3=CFG["max_cdr3"],
            max_pep=CFG["max_pep"],
            max_mhc=CFG["max_mhc"],
        ).to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

        best_score = -1e9
        best_epoch = -1
        bad_count = 0
        history = []
        ckpt_path = os.path.join(fold_dir, "best_model.pt")

        for epoch in range(1, CFG["epochs"] + 1):
            t0 = time.time()
            tr_loss = train_one_epoch(model, tr_loader, optimizer, device, loss_fn, grad_clip=CFG["grad_clip"])
            va_metrics = evaluate(model, va_loader, device, return_arrays=False)

            mon = CFG["monitor"]
            score = float(va_metrics.get(mon, np.nan))
            if math.isnan(score):
                score = -1e9

            improved = (score > best_score + CFG["min_delta"])
            if improved:
                best_score = score
                best_epoch = epoch
                bad_count = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": CFG,
                        "train_variant": train_variant,
                        "fold": fold_id,
                        "model_name": "AttBaselineModel",
                    },
                    ckpt_path
                )
            else:
                bad_count += 1

            dt = time.time() - t0
            history.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_metrics": va_metrics,
                "monitor": mon,
                "monitor_score": score,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "time_sec": dt
            })

            print(f"[{train_variant}][fold{fold_id}] ep{epoch:02d} loss={tr_loss:.4f} "
                  f"val_{mon}={score:.4f} best={best_score:.4f}(ep{best_epoch}) bad={bad_count}/{CFG['patience']} {dt:.1f}s")

            if bad_count >= CFG["patience"]:
                break

        save_json(history, os.path.join(fold_dir, "train_history.json"))

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        cross_eval = {}
        for eval_variant in CFG["variants"]:
            eval_dir = os.path.join(fold_dir, f"eval_on_{eval_variant}")
            _ensure_dir(eval_dir)
            print(f"  [EVAL] train={train_variant} fold={fold_id} -> eval={eval_variant}")
            cross_eval[eval_variant] = evaluate_on_variant(model, eval_variant, device, eval_dir)

        fold_results[f"fold_{fold_id}"] = {
            "best_epoch": best_epoch,
            "best_val_score": best_score,
            "cross_eval": cross_eval,
            "n_train": int(len(df_tr)),
            "n_val": int(len(df_va)),
        }

    return fold_results


def aggregate_cv(all_cv: dict):
    """
    Aggregate per-fold metrics:
      agg[train_variant][eval_variant][split] = {ROC_AUC: {mean,std,n}, PR_AUC: {mean,std,n}}
    """
    agg = {}

    for train_variant, fold_map in all_cv.items():
        agg.setdefault(train_variant, {})

        eval_variants = set()
        for _, fold_res in fold_map.items():
            eval_variants |= set(fold_res["cross_eval"].keys())

        for eval_variant in sorted(eval_variants):
            agg[train_variant].setdefault(eval_variant, {})

            split_names = set()
            for _, fold_res in fold_map.items():
                split_names |= set(fold_res["cross_eval"][eval_variant].keys())

            for split_name in sorted(split_names):
                roc_list, pr_list = [], []
                for _, fold_res in fold_map.items():
                    m = fold_res["cross_eval"][eval_variant].get(split_name, {})
                    if "ROC_AUC" in m and m["ROC_AUC"] is not None and not (isinstance(m["ROC_AUC"], float) and np.isnan(m["ROC_AUC"])):
                        roc_list.append(float(m["ROC_AUC"]))
                    if "PR_AUC" in m and m["PR_AUC"] is not None and not (isinstance(m["PR_AUC"], float) and np.isnan(m["PR_AUC"])):
                        pr_list.append(float(m["PR_AUC"]))

                def _ms(x):
                    if len(x) == 0:
                        return {"mean": np.nan, "std": np.nan, "n": 0}
                    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1) if len(x) > 1 else 0.0), "n": int(len(x))}

                agg[train_variant][eval_variant][split_name] = {
                    "ROC_AUC": _ms(roc_list),
                    "PR_AUC": _ms(pr_list),
                }

    return agg


def main():
    _ensure_dir(CFG["out_root"])
    save_json(CFG, os.path.join(CFG["out_root"], "config_baseline.json"))

    all_cv = {}
    for train_variant in RUN_VARIANTS:
        print(f"\n==============================")
        print(f"Run 5-fold CV (baseline) for ratio: {train_variant}")
        print(f"==============================")
        all_cv[train_variant] = run_train_variant_5fold(train_variant)

    save_json(all_cv, os.path.join(CFG["out_root"], "summary_cv_baseline.json"))

    agg = aggregate_cv(all_cv)
    save_json(agg, os.path.join(CFG["out_root"], "summary_agg_baseline.json"))


if __name__ == "__main__":
    main()
