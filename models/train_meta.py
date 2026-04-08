# -*- coding: utf-8 -*-
"""
train_exp2_meta_hybrid_strict_blind.py

混合系统（Hybrid System） - 完全盲测版 (Strict Blind)
功能：
1. 严禁在训练/模型选择过程中加载 Unseen Test Set。
2. 使用 Meta-Train 内部划分出的 Hold-out Set 作为“防遗忘监控”。
3. 依靠正则化 (Reg) 和 低学习率 (Low LR) 保证稳定性。
"""

import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from sklearn.model_selection import train_test_split # 新增

# 复用模块
from dataset import PMHCTCRDataset, preprocess
from model_baseline import AttBaselineModel, set_seed, evaluate, _compute_metrics_from_arrays

# =========================
# Config
# =========================
NEG_RATIO = "1000x" 
BASE_DIR = f"data/splits_meta_with_neg/exp2_healthy_{NEG_RATIO}"
BASELINE_ROOT = f"result/exp2_cross_5fold_baseline/train_exp2_healthy_{NEG_RATIO}"

@dataclass
class CFG:
    # 路径配置
    meta_train_csv: str = f"{BASE_DIR}/meta_train.csv"
    
    # 【注意】：Unseen Test 仅用于最终汇报，绝不参与训练过程
    unseen_test_csv: str = f"{BASE_DIR}/unseen_test.csv"
    
    meta_test_csv: str = f"{BASE_DIR}/meta_test.csv"
    few_support_csv: str = f"{BASE_DIR}/few_test_support.csv"
    few_query_csv: str = f"{BASE_DIR}/few_test_query.csv"
    
    baseline_fold_dir: str = f"{BASELINE_ROOT}/fold_1" 
    out_root: str = f"result/meta_{NEG_RATIO}_5fold" # 修改输出目录
    
    # Meta Training Params
    n_folds: int = 5
    seed: int = 2025
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 5
    
    # Outer Loop
    outer_lr: float = 0.1       # 保持低 LR
    outer_lr_decay: float = 0.90
    
    # Inner Loop
    inner_steps: int = 3       
    inner_lr: float = 1e-3
    inner_batch_size: int = 32
    train_use_reg: bool = True  # 保持正则化开启
    
    # TTFT Params
    ft_steps: int = 20         
    ft_lr: float = 0.002       
    reg_lambda: float = 0.5    
    
    # Model Config
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 1
    ffn_dim: int = 512
    dropout: int = 0.1
    max_cdr3: int = 60
    max_pep: int = 25
    max_mhc: int = 128
    task_col: str = "task_id"

# =========================
# Utils (保持不变)
# =========================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_csv(path):
    return pd.read_csv(path)

def get_grouped_tasks(df, task_col="task_id"):
    groups = df.groupby(task_col)
    task_map = {k: v.reset_index(drop=True) for k, v in groups}
    return task_map, list(task_map.keys())

def build_loader_from_df(df, batch_size, shuffle=True):
    ds = PMHCTCRDataset(df, max_cdr3=CFG.max_cdr3, max_pep=CFG.max_pep, max_mhc=CFG.max_mhc)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

def set_strict_seed_for_eval(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Core: Regularized Update (保持不变)
# =========================
def train_on_task_support(model, df_support, steps, lr, device, use_reg=False, strict_mode=False):
    fmodel = copy.deepcopy(model)
    fmodel.eval() 
    
    anchor_params = {}
    if use_reg:
        for name, param in fmodel.named_parameters():
            if param.requires_grad:
                anchor_params[name] = param.clone().detach()

    params_to_opt = [p for p in fmodel.parameters() if p.requires_grad]
    if not params_to_opt: params_to_opt = fmodel.parameters()
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    
    if strict_mode:
        bs = len(df_support)
        shuffle = False
    else:
        bs = min(len(df_support), CFG.inner_batch_size)
        if bs < 2: bs = 2
        shuffle = True
    
    loader = build_loader_from_df(df_support, bs, shuffle=shuffle)
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
        
        if use_reg:
            l2_loss = 0.0
            for name, param in fmodel.named_parameters():
                if param.requires_grad:
                    l2_loss += torch.norm(param - anchor_params[name]) ** 2
            total_loss = bce_loss + (CFG.reg_lambda * l2_loss)
        else:
            total_loss = bce_loss
            
        total_loss.backward()
        optimizer.step()
        
    return fmodel

# =========================
# Evaluation Logic (保持不变)
# =========================
def eval_general(model, df, device):
    model.eval()
    loader = build_loader_from_df(df, 256, shuffle=False)
    metrics = evaluate(model, loader, device, return_arrays=False)
    return metrics

def eval_few_shot_with_ttft(model, df_support, df_query, device, task_col, strict_mode=False):
    supp_map, tasks_s = get_grouped_tasks(df_support, task_col)
    query_map, tasks_q = get_grouped_tasks(df_query, task_col)
    common_tasks = sorted(list(set(tasks_s).intersection(tasks_q)))
    
    all_y_true = []
    all_y_prob = []
    
    for i, t_id in enumerate(common_tasks):
        sub_supp = supp_map[t_id]
        sub_query = query_map[t_id]
        
        if strict_mode:
            set_strict_seed_for_eval(2025 + i)
        
        ft_model = train_on_task_support(model, sub_supp, 
                                         steps=CFG.ft_steps, 
                                         lr=CFG.ft_lr, 
                                         device=device,
                                         use_reg=True,
                                         strict_mode=strict_mode)
        
        ft_model.eval()
        q_loader = build_loader_from_df(sub_query, 256, shuffle=False)
        _, y_t, y_p = evaluate(ft_model, q_loader, device, return_arrays=True)
        all_y_true.append(y_t)
        all_y_prob.append(y_p)
        
    if not all_y_true: return {}
    return _compute_metrics_from_arrays(np.concatenate(all_y_true), np.concatenate(all_y_prob))

# =========================
# Main Pipeline (逻辑重构)
# =========================

def run_hybrid_fold_blind(fold_id, train_df_fold, val_df_fold, device):
    """
    Blind Training:
    不传入 unseen_test，而是依赖 val_df_fold (Meta-Val) 进行模型选择。
    通过 Low LR + Reg 来隐式地保护 Unseen 性能。
    """
    print(f"\n>>> Starting Blind Hybrid Training for Fold {fold_id} <<<")
    
    # 1. Initialize
    model = AttBaselineModel(
        d_model=CFG.d_model, nhead=CFG.nhead, num_layers=CFG.num_layers,
        ffn_dim=CFG.ffn_dim, attn_dropout=CFG.dropout, emb_dropout=CFG.dropout,
        max_cdr3=CFG.max_cdr3, max_pep=CFG.max_pep, max_mhc=CFG.max_mhc
    ).to(device)
    
    # 2. Warm Start
    ckpt_path = f"{CFG.baseline_fold_dir.replace('fold_1', f'fold_{fold_id}')}/best_model.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading baseline weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        print(f"Warning: Baseline checkpoint not found.")

    # 3. Partial Freeze
    for param in model.parameters(): param.requires_grad = False
    trainable_params = []
    head_keywords = ["mlp", "out", "classifier", "head", "fc"]
    target_layer_idx = str(CFG.num_layers - 1)
    
    for name, param in model.named_parameters():
        should_train = False
        if any(k in name for k in head_keywords):
            if "enc_" not in name and "transformer" not in name: should_train = True
        if "layers" in name and f".{target_layer_idx}." in name: should_train = True
        if "norm" in name and "enc" in name: should_train = True

        if should_train:
            param.requires_grad = True
            trainable_params.append(name)
    
    task_map, task_ids = get_grouped_tasks(train_df_fold, CFG.task_col)

    # 4. Reptile Training Loop
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    curr_outer_lr = CFG.outer_lr
    
    # 计算初始的 Validation Performance (Meta-Val)
    print("  Checking Baseline Meta-Val Performance...")
    # 注意：这里我们只看 val_df_fold，这是 Meta-Train 内部划分出来的验证集
    base_val_res = eval_general(model, val_df_fold, device)
    print(f"  [Baseline] Meta-Val PR_AUC: {base_val_res['PR_AUC']:.4f}")

    for epoch in range(1, CFG.epochs + 1):
        np.random.shuffle(task_ids)
        theta_old_global = {k: v.clone() for k, v in model.state_dict().items()}
        
        for t_id in task_ids:
            df_task = task_map[t_id]
            
            # Inner Loop (with Reg)
            ft_model = train_on_task_support(model, df_task, 
                                             steps=CFG.inner_steps, 
                                             lr=CFG.inner_lr, 
                                             device=device,
                                             use_reg=CFG.train_use_reg, 
                                             strict_mode=False) 
            
            # Outer Update
            theta_new = ft_model.state_dict()
            new_state = model.state_dict()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        diff = theta_new[name] - theta_old_global[name]
                        new_state[name] = theta_old_global[name] + curr_outer_lr * diff
            model.load_state_dict(new_state)
            
        curr_outer_lr *= CFG.outer_lr_decay
        
        # --- Model Selection (Blind) ---
        # 我们只根据 Meta-Val (val_df_fold) 来选择模型
        # 如果模型在这里过拟合，Reg 和 Low LR 应该会阻止它在 Unseen 上崩太厉害
        val_metrics = eval_general(model, val_df_fold, device)
        val_pr = val_metrics["PR_AUC"]
        
        print(f"  [Fold {fold_id} Ep {epoch}] OuterLR={curr_outer_lr:.4f} | Meta-Val PR={val_pr:.4f}")
        
        if val_pr > best_score:
            best_score = val_pr
            best_state = copy.deepcopy(model.state_dict())
            print("    >>> New Best Model Saved (based on Meta-Val)!")

    print(f"Fold {fold_id} Done. Best Meta-Val Score: {best_score:.4f}")
    model.load_state_dict(best_state)
    return model

def main():
    set_seed(CFG.seed)
    _ensure_dir(CFG.out_root)
    
    print("Loading datasets...")
    df_meta_train = preprocess(load_csv(CFG.meta_train_csv))
    
    # 真实 Unseen Test，仅用于最后一步
    df_unseen_test = preprocess(load_csv(CFG.unseen_test_csv)) 
    
    df_meta_test = preprocess(load_csv(CFG.meta_test_csv))
    df_few_supp = preprocess(load_csv(CFG.few_support_csv))
    df_few_query = preprocess(load_csv(CFG.few_query_csv))
    
    from train_exp2_baseline import make_group_folds
    # 这里我们使用标准的 K-Fold，每一折会自动产生 train_idx 和 val_idx
    # val_idx 对应的数据就是我们的 "Hold-out"，用于选择模型
    folds = make_group_folds(df_meta_train, CFG.task_col, CFG.n_folds, CFG.seed)
    
    results_agg = []
    
    for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
        df_tr = df_meta_train.iloc[tr_idx].reset_index(drop=True)
        df_va = df_meta_train.iloc[va_idx].reset_index(drop=True) # 这就是我们的 "Validation Set"
        
        # 【关键】：这里不再传入 df_unseen_monitor
        meta_model = run_hybrid_fold_blind(fold_id, df_tr, df_va, CFG.device)
        
        # --- 最终评估阶段 ---
        # 只有在这里，模型训练结束后，我们才拿 Unseen Test 来做一次“考试”
        print(f"Fold {fold_id} Final Evaluation...")
        res = {"fold": fold_id}
        
        # 1. Unseen Test (True Blind Test)
        m_unseen = eval_general(meta_model, df_unseen_test, CFG.device)
        res["unseen_ROC"] = m_unseen["ROC_AUC"]
        res["unseen_PR"] = m_unseen["PR_AUC"]
        
        # 2. Meta Test
        m_meta = eval_general(meta_model, df_meta_test, CFG.device)
        res["meta_test_ROC"] = m_meta["ROC_AUC"]
        res["meta_test_PR"] = m_meta["PR_AUC"]
        
        # 3. Few Shot Tasks
        print("  Evaluating Few-Shot Tasks...")
        m_few = eval_few_shot_with_ttft(meta_model, df_few_supp, df_few_query, 
                                        CFG.device, CFG.task_col, 
                                        strict_mode=True)
        res["few_query_ROC"] = m_few.get("ROC_AUC", 0)
        res["few_query_PR"] = m_few.get("PR_AUC", 0)
        
        print(f"Fold {fold_id} Result Summary:")
        print(f"  Unseen (Blind): {res['unseen_PR']:.4f}")
        print(f"  MetaTest      : {res['meta_test_PR']:.4f}")
        print(f"  FewQuery      : {res['few_query_PR']:.4f}")
        
        results_agg.append(res)
        torch.save(meta_model.state_dict(), f"{CFG.out_root}/model_fold_{fold_id}.pt")

    df_res = pd.DataFrame(results_agg)
    df_res.loc["mean"] = df_res.mean()
    df_res.to_csv(f"{CFG.out_root}/results_summary.csv")
    print("\n=== Final Results Summary (Blind Training) ===")
    print(df_res)

if __name__ == "__main__":
    main()