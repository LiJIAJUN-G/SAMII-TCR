# -*- coding: utf-8 -*-
"""
predict_hybrid_final_cmd.py

混合系统综合预测框架 - 5折集成版 (5-Fold Ensemble)
支持命令行参数输入，默认参数为预设路径。

功能：
1. 自动区分 Seen Task (Baseline) 和 Unseen Task (Meta)。
2. 自动检测是否有 Support 数据用于 Meta TTFT。
3. 核心策略：所有预测均使用 5 个模型的概率均值 (Ensemble Averaging)。
"""

import os
import copy
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 复用您提供的模块 (确保 dataset.py 和 model_baseline.py 在同一目录下)
from dataset import PMHCTCRDataset, preprocess
from model_baseline import AttBaselineModel, _compute_metrics_from_arrays

# =========================
# Default Defaults (Hardcoded)
# =========================
DEFAULT_META_TRAIN = "data/splits_meta_with_neg/exp2_healthy_100x/meta_train.csv"
DEFAULT_TEST_FILE = "data/splits_meta_with_neg/exp2_healthy_100x/unseen_test.csv"
DEFAULT_SUPPORT_FILE = "data/splits_meta_with_neg/exp2_healthy_100x/few_test_support.csv"
DEFAULT_OUTPUT = "result/final_prediction_result.csv"
DEFAULT_BASELINE_DIR = "result/exp2_cross_5fold_baseline/train_exp2_healthy_10x"
DEFAULT_META_DIR = "result/meta_10x_5fold"

# =========================
# Configuration Class
# =========================
class CFG:
    # 路径配置 (将被命令行参数覆盖)
    meta_train_path = DEFAULT_META_TRAIN
    test_file_path = DEFAULT_TEST_FILE
    support_file_path = DEFAULT_SUPPORT_FILE
    output_path = DEFAULT_OUTPUT
    baseline_model_dir = DEFAULT_BASELINE_DIR
    meta_model_dir = DEFAULT_META_DIR
    
    # TTFT 微调参数
    ft_steps = 20         
    ft_lr = 0.002         
    reg_lambda = 0.5      
    
    # 模型架构参数 (必须与训练一致)
    d_model = 256
    nhead = 8
    num_layers = 1
    ffn_dim = 512
    dropout = 0.1
    max_cdr3 = 60
    max_pep = 25
    max_mhc = 128
    
    # 推理参数
    batch_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = 5

# =========================
# Utils
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid System Prediction with 5-Fold Ensemble & TTFT")
    
    parser.add_argument("--meta_train", type=str, default=DEFAULT_META_TRAIN, 
                        help="Path to meta_train.csv (used to identify SEEN tasks)")
    parser.add_argument("--test_file", type=str, default=DEFAULT_TEST_FILE, 
                        help="Path to the test/query CSV file")
    parser.add_argument("--support_file", type=str,  
                        help="Path to the support CSV file for TTFT (optional). If not provided, uses Zero-shot.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, 
                        help="Path to save the final prediction CSV")
    
    parser.add_argument("--baseline_dir", type=str, default=DEFAULT_BASELINE_DIR, 
                        help="Directory containing 5-fold Baseline models")
    parser.add_argument("--meta_dir", type=str, default=DEFAULT_META_DIR, 
                        help="Directory containing 5-fold Meta models")
    
    parser.add_argument("--device", type=str, default=CFG.device, 
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size")
    
    return parser.parse_args()

def _fill_na_for_task(x):
    return "NA" if pd.isna(x) else str(x)

def add_task_id(df):
    """生成 Task ID 以便进行路由"""
    req_cols = ["epitope", "MHCA", "MHCB"]
    for c in req_cols:
        if c not in df.columns:
            df[c] = np.nan
    df["task_id"] = (
        df["epitope"].map(_fill_na_for_task) + "|" +
        df["MHCA"].map(_fill_na_for_task) + "|" +
        df["MHCB"].map(_fill_na_for_task)
    )
    return df

def build_loader(df, shuffle=False):
    ds = PMHCTCRDataset(df, max_cdr3=CFG.max_cdr3, max_pep=CFG.max_pep, max_mhc=CFG.max_mhc)
    return DataLoader(ds, batch_size=CFG.batch_size, shuffle=shuffle, num_workers=0)

def load_ensemble_models(root_dir, style="baseline"):
    """
    加载 5 个模型并返回列表
    style: 'baseline' (ckpt['model']) 或 'meta' (直接 state_dict)
    """
    models = []
    print(f"Loading {style} ensemble ({CFG.n_folds} folds) from {root_dir}...")
    
    for fold in range(1, CFG.n_folds + 1):
        # 兼容两种文件命名习惯
        path_v1 = os.path.join(root_dir, f"fold_{fold}", "best_model.pt") # Baseline
        path_v2 = os.path.join(root_dir, f"model_fold_{fold}.pt")        # Meta
        
        if os.path.exists(path_v1):
            path = path_v1
        elif os.path.exists(path_v2):
            path = path_v2
        else:
            print(f"  [Warning] Fold {fold} not found at {path_v1} or {path_v2}. Skipping.")
            continue
            
        # 初始化模型
        model = AttBaselineModel(
            d_model=CFG.d_model, nhead=CFG.nhead, num_layers=CFG.num_layers,
            ffn_dim=CFG.ffn_dim, attn_dropout=CFG.dropout, emb_dropout=CFG.dropout,
            max_cdr3=CFG.max_cdr3, max_pep=CFG.max_pep, max_mhc=CFG.max_mhc
        ).to(CFG.device)
        
        # 加载权重
        try:
            ckpt = torch.load(path, map_location=CFG.device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"  [Error] Failed to load fold {fold}: {e}")

    print(f"  Loaded {len(models)} models.")
    return models

# =========================
# Core Logic: TTFT Training
# =========================

def train_on_task_support(model, df_support, steps, lr, device):
    """
    Test-Time Fine-Tuning: 在 Support Set 上微调单个模型
    """
    fmodel = copy.deepcopy(model)
    fmodel.eval() # 保持 BN/Dropout 为评估模式
    
    # 1. 锚点参数 (用于正则化，防止遗忘)
    anchor_params = {}
    for name, param in fmodel.named_parameters():
        if param.requires_grad:
            anchor_params[name] = param.clone().detach()

    # 2. 优化器
    params_to_opt = [p for p in fmodel.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Full Batch Training
    bs = len(df_support)
    if bs < 2 and len(df_support) > 0: bs = len(df_support) 
    
    loader = DataLoader(
        PMHCTCRDataset(df_support, CFG.max_cdr3, CFG.max_pep, CFG.max_mhc),
        batch_size=bs, shuffle=True
    )
    iterator = iter(loader)
    
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
            
        for k in ["cdr3", "pep", "mhc", "y"]:
            batch[k] = batch[k].to(device)
            
        optimizer.zero_grad()
        logit, _ = fmodel(batch)
        bce_loss = loss_fn(logit, batch["y"])
        
        # L2 Regularization
        l2_loss = 0.0
        for name, param in fmodel.named_parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param - anchor_params[name]) ** 2
        
        total_loss = bce_loss + (CFG.reg_lambda * l2_loss)
        total_loss.backward()
        optimizer.step()
        
    return fmodel

# =========================
# Prediction Logic
# =========================

def predict_ensemble_simple(models, df):
    """简单集成：所有模型预测取平均"""
    if len(df) == 0: return np.array([])
    
    loader = build_loader(df, shuffle=False)
    all_preds = []
    
    with torch.no_grad():
        for model in models:
            preds_fold = []
            for b in loader:
                for k in ["cdr3","pep","mhc"]: b[k] = b[k].to(CFG.device)
                logit, _ = model(b)
                prob = torch.sigmoid(logit).cpu().numpy()
                preds_fold.extend(prob)
            all_preds.append(preds_fold)
            
    # 对 5 个模型的预测结果取平均
    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds

def predict_ensemble_ttft(models, df_query, df_support):
    """
    TTFT 集成：按 Task 分组 -> 5折微调 -> 预测 -> 平均
    """
    tasks = df_query["task_id"].unique()
    supp_groups = {k: v for k, v in df_support.groupby("task_id")}
    query_groups = {k: v for k, v in df_query.groupby("task_id")}
    
    final_preds = pd.Series(index=df_query.index, dtype=float)
    
    print(f"  [TTFT] Processing {len(tasks)} tasks with support data...")
    
    for t_id in tqdm(tasks):
        sub_query = query_groups[t_id]
        sub_supp = supp_groups.get(t_id, None)
        
        if sub_supp is None or len(sub_supp) == 0:
            # 回退到 Zero-shot
            batch_preds = predict_ensemble_simple(models, sub_query)
        else:
            # === 5-Fold TTFT ===
            fold_preds = []
            for model in models:
                # 微调
                ft_model = train_on_task_support(
                    model, sub_supp, 
                    steps=CFG.ft_steps, 
                    lr=CFG.ft_lr, 
                    device=CFG.device
                )
                
                # 预测
                ft_model.eval()
                loader = build_loader(sub_query, shuffle=False)
                p_list = []
                with torch.no_grad():
                    for b in loader:
                        for k in ["cdr3","pep","mhc"]: b[k] = b[k].to(CFG.device)
                        logit, _ = ft_model(b)
                        p_list.extend(torch.sigmoid(logit).cpu().numpy())
                fold_preds.append(p_list)
            
            # 平均
            batch_preds = np.mean(fold_preds, axis=0)
            
        final_preds.loc[sub_query.index] = batch_preds
        
    return final_preds.values

# =========================
# Main Pipeline
# =========================

def main():
    # 1. Parse Arguments & Update Config
    args = parse_args()
    CFG.meta_train_path = args.meta_train
    CFG.test_file_path = args.test_file
    CFG.support_file_path = args.support_file
    CFG.output_path = args.output
    CFG.baseline_model_dir = args.baseline_dir
    CFG.meta_model_dir = args.meta_dir
    CFG.device = args.device
    CFG.batch_size = args.batch_size
    
    print("=== Hybrid System Final Prediction (5-Fold Ensemble) ===")
    print(f"Input:    {CFG.test_file_path}")
    print(f"Support:  {CFG.support_file_path if CFG.support_file_path else 'None'}")
    print(f"Output:   {CFG.output_path}")

    # 2. 准备任务信息
    print(f"\nStep 1: Analyzing Meta-Train tasks (Seen Tasks)...")
    if os.path.exists(CFG.meta_train_path):
        # --- 修正开始 ---
        print("  -> Loading and Preprocessing Meta-Train...")
        df_train_raw = pd.read_csv(CFG.meta_train_path)
        # 必须使用 preprocess 保证 epitope/MHCA/MHCB 的格式与测试集完全一致
        df_train = preprocess(df_train_raw)
        df_train = add_task_id(df_train)
        # --- 修正结束 ---
        
        seen_tasks = set(df_train["task_id"].unique())
        print(f"  -> Found {len(seen_tasks)} seen tasks.")
    else:
        print(f"  [Warning] Meta-Train file not found at {CFG.meta_train_path}. Assuming ALL tasks are Unseen.")
        seen_tasks = set()

    # 3. 准备测试数据 (保持不变)
    print(f"\nStep 2: Loading Test Data...")
    if not os.path.exists(CFG.test_file_path):
        raise FileNotFoundError(f"Test file not found: {CFG.test_file_path}")
        
    if CFG.test_file_path.endswith('.pkl'):
        print(f"  -> Loading Pickle file (Fast Mode): {CFG.test_file_path}")
        df_raw = pd.read_pickle(CFG.test_file_path)
    else:
        df_raw = pd.read_csv(CFG.test_file_path)
        
    df_test = preprocess(df_raw)
    df_test = add_task_id(df_test)
    
    # 4. 准备 Support 数据 (可选)
    df_support = pd.DataFrame()
    support_tasks = set()
    if CFG.support_file_path and os.path.exists(CFG.support_file_path):
        print(f"  -> Loading Support Data from {CFG.support_file_path}")
        df_support = preprocess(pd.read_csv(CFG.support_file_path))
        df_support = add_task_id(df_support)
        support_tasks = set(df_support["task_id"].unique())
    else:
        print("  -> No Support Data provided/found. TTFT will be skipped.")

    # 5. 路由策略 (Routing)
    mask_seen = df_test["task_id"].isin(seen_tasks)
    mask_ttft = (~mask_seen) & (df_test["task_id"].isin(support_tasks))
    mask_zero = (~mask_seen) & (~mask_ttft)
    
    df_seen = df_test[mask_seen].copy()
    df_ttft = df_test[mask_ttft].copy()
    df_zero = df_test[mask_zero].copy()
    
    print("\nProcessing Strategy:")
    print(f"  1. [Baseline Ensemble] (Seen Tasks):       {len(df_seen)} samples")
    print(f"  2. [Meta TTFT Ensemble] (Unseen+Support):  {len(df_ttft)} samples")
    print(f"  3. [Meta Zero Ensemble] (Unseen Only):     {len(df_zero)} samples")
    
    final_results = pd.Series(index=df_test.index, dtype=float)
    
    # --- 执行 Baseline 分支 ---
    if len(df_seen) > 0:
        print("\n>>> Running Baseline Ensemble...")
        base_models = load_ensemble_models(CFG.baseline_model_dir, "baseline")
        if base_models:
            preds = predict_ensemble_simple(base_models, df_seen)
            final_results.loc[df_seen.index] = preds
            del base_models; torch.cuda.empty_cache()
        else:
            print("  [Error] No baseline models loaded. Skipping seen tasks.")
    
    # --- 执行 Meta 分支 ---
    meta_models = []
    if len(df_ttft) > 0 or len(df_zero) > 0:
        print("\n>>> Loading Meta Models...")
        meta_models = load_ensemble_models(CFG.meta_model_dir, "meta")
    
    if len(df_ttft) > 0 and meta_models:
        print("\n>>> Running Meta TTFT Ensemble (Fine-tuning)...")
        rel_tasks = df_ttft["task_id"].unique()
        df_supp_rel = df_support[df_support["task_id"].isin(rel_tasks)]
        
        preds = predict_ensemble_ttft(meta_models, df_ttft, df_supp_rel)
        final_results.loc[df_ttft.index] = preds
        
    if len(df_zero) > 0 and meta_models:
        print("\n>>> Running Meta Zero-shot Ensemble...")
        preds = predict_ensemble_simple(meta_models, df_zero)
        final_results.loc[df_zero.index] = preds
        
    # 6. 保存与评估
    print("\nStep 3: Saving results...")
    df_test["hybrid_prob"] = final_results.values
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(CFG.output_path)), exist_ok=True)
    df_test.to_csv(CFG.output_path, index=False)
    print(f"  -> Saved to {CFG.output_path}")
    
    if "label" in df_test.columns:
        print("\n=== Performance Report ===")
        mask_valid = ~df_test["hybrid_prob"].isna()
        if mask_valid.sum() == 0:
            print("  No valid predictions to evaluate.")
        else:
            y_true = df_test.loc[mask_valid, "label"].values
            y_prob = df_test.loc[mask_valid, "hybrid_prob"].values
            metrics = _compute_metrics_from_arrays(y_true, y_prob)
            print(f"  Samples Evaluated: {len(y_true)}")
            print(f"  ROC_AUC: {metrics.get('ROC_AUC', 0):.4f}")
            print(f"  PR_AUC:  {metrics.get('PR_AUC', 0):.4f}")
            print(f"  Acc:     {metrics.get('Acc', 0):.4f}")

if __name__ == "__main__":
    main()