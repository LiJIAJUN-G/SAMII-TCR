# -*- coding: utf-8 -*-
"""
train_exp1_cross_5fold_improved.py

实验一：使用改进的模型进行交叉训练-测试 + 5折交叉验证
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
# ✅ 导入改进的模型
from model_baseline import AttBaselineModel, set_seed, evaluate

# =========================
# Config
# =========================
CFG = {
    "seed": 2025,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 数据根目录：s2_gen_neg.py 输出
    "data_root": "data/splits_meta_with_neg",

    # 两种负样本方案（训练/测试都会用）
    "variants": ["exp1_shuffle", "exp1_healthy"],

    # 训练 pool（做5折）
    "train_file": "meta_train.csv",

    # 测试集（三个都测）
    "test_files": {
        "meta_test": "meta_test.csv",
        "few_test_query": "few_test_query.csv",
        "unseen_test": "unseen_test.csv",
    },

    # 可选 merged test over 3 splits
    "also_eval_merged_test": True,

    # meta_train 上的 task 列名（来自 s1）
    "task_col": "task_id",

    # 5fold
    "n_folds": 5,

    # DataLoader
    "batch_size": 256,
    "num_workers": 4,
    "pin_memory": True,

    # ✅ 改进模型的配置
    "d_model": 256,
    "nhead": 8,
    "num_layers": 1,
    "ffn_dim": 512,
    "dropout": 0.1,
    "max_cdr3": 60,
    "max_pep": 25,
    "max_mhc": 128,

    # ✅ 改进模型的训练参数
    "epochs": 30,
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,

    # Early stopping on fold-val
    "monitor": "PR_AUC",      # "PR_AUC" or "ROC_AUC"
    "patience": 5,
    "min_delta": 1e-4,

    # Output
    "out_root": "result/exp1_cross_5fold_baseline",  # ✅ 修改输出目录
}


# =========================
# Utils (保持不变)
# =========================
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
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
        logit, aux = model(b)  # ✅ 接收aux输出（虽然不使用）
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
    with open(os.path.join(out_dir, f"test_metrics_{split_name}.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}\t{v}\n")
    return metrics

def evaluate_on_variant(model, eval_variant: str, device, out_dir: str):
    """
    在某个测试负样本方案（eval_variant）的 test splits 上分别评估并保存。
    """
    data_dir = os.path.join(CFG["data_root"], eval_variant)

    results = {}
    for split_name, file_name in CFG["test_files"].items():
        p = os.path.join(data_dir, file_name)
        df_raw = load_csv(p)
        results[split_name] = eval_one_split(model, df_raw, device, split_name, out_dir)

    if CFG["also_eval_merged_test"]:
        dfs = []
        for _, file_name in CFG["test_files"].items():
            dfs.append(load_csv(os.path.join(data_dir, file_name)))
        merged_raw = pd.concat(dfs, axis=0, ignore_index=True)
        results["merged_all_tests"] = eval_one_split(model, merged_raw, device, "merged_all_tests", out_dir)

    save_json(results, os.path.join(out_dir, "all_test_results.json"))
    return results

def make_group_folds(df: pd.DataFrame, group_col: str, n_splits: int, seed: int):
    """
    生成按 group_col 分组的 folds（同一 task 不会跨fold）。
    返回 list[(train_idx, val_idx)]，idx 是 df 的行索引（0..len-1）。
    """
    if group_col not in df.columns:
        raise ValueError(f"meta_train 缺少分组列 {group_col}，请确认 s1 产生了 task_id 并保留到后续文件。")

    groups = df[group_col].astype(str).values
    uniq = np.unique(groups)

    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    # round-robin 分配 task 到 folds
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


# =========================
# 改进的训练函数
# =========================
def run_train_variant_5fold(train_variant: str):
    set_seed(CFG["seed"])
    device = torch.device(CFG["device"])

    train_data_dir = os.path.join(CFG["data_root"], train_variant)
    out_base = os.path.join(CFG["out_root"], f"train_{train_variant}")
    _ensure_dir(out_base)

    # ---- Load meta_train (pool for CV) ----
    df_train_raw_all = load_csv(os.path.join(train_data_dir, CFG["train_file"]))
    df_train_all = preprocess(df_train_raw_all)

    # ✅ 数据验证
    print(f"[DATA] {train_variant}: {len(df_train_all)} samples after preprocessing")
    print(f"[DATA] Positive ratio: {df_train_all['label'].mean():.3f}")
    
    # 检查关键列
    for col in ['CDR3.beta.aa', 'MHC_pseudo', 'Epitope.peptide']:
        non_empty = (df_train_all[col].astype(str) != '').mean()
        print(f"[DATA] {col} non-empty ratio: {non_empty:.3f}")

    folds = make_group_folds(df_train_all, CFG["task_col"], CFG["n_folds"], CFG["seed"])

    fold_results = {}

    for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_dir = os.path.join(out_base, f"fold_{fold_id}")
        _ensure_dir(fold_dir)

        df_tr = df_train_all.iloc[tr_idx].reset_index(drop=True)
        df_va = df_train_all.iloc[va_idx].reset_index(drop=True)

        # ✅ 检查验证集平衡性
        val_pos_ratio = df_va['label'].mean()
        print(f"[FOLD {fold_id}] Train: {len(df_tr)}, Val: {len(df_va)}, Val positive ratio: {val_pos_ratio:.3f}")

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

        # ✅ 使用改进的模型
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

        # ✅ 模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        
        # ✅ 修复：移除verbose参数
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

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

            # ✅ 更新学习率
            scheduler.step(score)

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
                        "epoch": epoch,
                        "val_score": score
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

            # ✅ 添加学习率信息
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{train_variant}][fold{fold_id}] ep{epoch:02d} loss={tr_loss:.4f} "
                  f"val_{mon}={score:.4f} best={best_score:.4f}(ep{best_epoch}) "
                  f"bad={bad_count}/{CFG['patience']} lr={current_lr:.2e} {dt:.1f}s")

            if bad_count >= CFG["patience"]:
                print(f"[EARLY STOP] Early stopping at epoch {epoch}")
                break

        save_json(history, os.path.join(fold_dir, "train_history.json"))

        # ---- Load best and cross-eval on BOTH variants ----
        print(f"[LOAD] Loading best model from epoch {best_epoch}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        cross_eval = {}
        for eval_variant in CFG["variants"]:
            eval_dir = os.path.join(fold_dir, f"eval_on_{eval_variant}")
            _ensure_dir(eval_dir)
            print(f"  [EVAL] train={train_variant} fold={fold_id} -> eval={eval_variant}")
            try:
                cross_eval[eval_variant] = evaluate_on_variant(model, eval_variant, device, eval_dir)
                print(f"  [EVAL SUCCESS] {eval_variant}")
            except Exception as e:
                print(f"  [EVAL ERROR] {eval_variant}: {e}")
                cross_eval[eval_variant] = {"error": str(e)}

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
    all_cv:
      {train_variant: {fold_k: {"cross_eval": {eval_variant: {split: metrics}}}}}
    返回聚合：
      agg[train_variant][eval_variant][split] = {ROC_AUC_mean/std, PR_AUC_mean/std, ...}
    """
    agg = {}

    for train_variant, fold_map in all_cv.items():
        agg.setdefault(train_variant, {})
        # 找到 eval_variant 集合
        eval_variants = set()
        for fold_id, fold_res in fold_map.items():
            eval_variants |= set(fold_res["cross_eval"].keys())

        for eval_variant in sorted(eval_variants):
            agg[train_variant].setdefault(eval_variant, {})
            # splits
            split_names = set()
            for fold_id, fold_res in fold_map.items():
                split_names |= set(fold_res["cross_eval"][eval_variant].keys())

            for split_name in sorted(split_names):
                # 收集所有指标
                metric_lists = {}
                for fold_id, fold_res in fold_map.items():
                    m = fold_res["cross_eval"][eval_variant].get(split_name, {})
                    if isinstance(m, dict) and "error" not in m:
                        for metric_name, metric_value in m.items():
                            if (isinstance(metric_value, (int, float)) and 
                                not np.isnan(metric_value) and metric_value != -1e9):
                                if metric_name not in metric_lists:
                                    metric_lists[metric_name] = []
                                metric_lists[metric_name].append(float(metric_value))

                def _ms(x):
                    if len(x) == 0:
                        return {"mean": np.nan, "std": np.nan, "n": 0}
                    return {
                        "mean": float(np.mean(x)), 
                        "std": float(np.std(x, ddof=1) if len(x) > 1 else 0.0), 
                        "n": int(len(x))
                    }

                agg[train_variant][eval_variant][split_name] = {
                    metric_name: _ms(values) for metric_name, values in metric_lists.items()
                }

    return agg


def write_paper_wide_table(agg: dict, out_csv: str):
    """
    论文友好宽表：
    行：train_variant × eval_variant
    列：meta_test / few_test_query / unseen_test 的 ROC_AUC、PR_AUC（mean/std）
    """
    splits = ["meta_test", "few_test_query", "unseen_test"]
    metrics = ["ROC_AUC", "PR_AUC"]

    rows = []
    for train_variant in sorted(agg.keys()):
        for eval_variant in sorted(agg[train_variant].keys()):
            row = {"train_variant": train_variant, "eval_variant": eval_variant}
            for sp in splits:
                for m in metrics:
                    ms = agg[train_variant][eval_variant].get(sp, {}).get(m, {"mean": np.nan, "std": np.nan, "n": 0})
                    row[f"{sp}_{m}_mean"] = ms["mean"]
                    row[f"{sp}_{m}_std"] = ms["std"]
                    row[f"{sp}_{m}_n"] = ms["n"]  # ✅ 添加样本数
            rows.append(row)

    df = pd.DataFrame(rows)

    # 固定列顺序
    cols = ["train_variant", "eval_variant"]
    for sp in splits:
        for m in metrics:
            cols += [f"{sp}_{m}_mean", f"{sp}_{m}_std", f"{sp}_{m}_n"]
    
    # 确保所有列都存在
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] saved paper-wide table: {out_csv}")
    return df


def main():
    _ensure_dir(CFG["out_root"])
    save_json(CFG, os.path.join(CFG["out_root"], "config.json"))

    all_cv = {}
    for train_variant in CFG["variants"]:
        print(f"\n{'='*60}")
        print(f"Run 5-fold CV with IMPROVED MODEL for train_variant: {train_variant}")
        print(f"{'='*60}")
        all_cv[train_variant] = run_train_variant_5fold(train_variant)

    # 保存结果
    save_json(all_cv, os.path.join(CFG["out_root"], "summary_cv.json"))
    
    # 聚合结果
    agg = aggregate_cv(all_cv)
    save_json(agg, os.path.join(CFG["out_root"], "summary_agg.json"))
    
    # 论文宽表
    wide_csv = os.path.join(CFG["out_root"], "summary_wide.csv")
    write_paper_wide_table(agg, wide_csv)
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"Results saved to: {CFG['out_root']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
