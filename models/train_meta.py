# -*- coding: utf-8 -*-
"""
Meta-training of SAMII-TCR with the hybrid Reptile-style outer loop.

Protocol:
  1. The unseen test set is never loaded during training or model selection.
  2. A hold-out fold carved from meta_train (task-disjoint) is used as the
     validation signal for model selection.
  3. Stability of the meta-update is enforced by a low outer learning rate and
     L2 anchoring (regularization toward the pre-update parameters) in the
     inner loop.

Each fold warm-starts from the corresponding supervised baseline fold
checkpoint, runs task-wise inner updates with a Reptile outer update, and is
finally evaluated on meta_test / unseen_test / few-shot query (with TTFT
adaptation on the few-shot support set).

Usage:
    NEG_RATIO=1000x python train_meta.py
    MODEL_SET=random_background NEG_RATIO=10x python train_meta.py
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

from dataset import PMHCTCRDataset, preprocess
from model import AttBaselineModel, set_seed, evaluate, _compute_metrics_from_arrays

NEG_RATIO = os.environ.get("NEG_RATIO", "1000x")
MODEL_SET = os.environ.get("MODEL_SET", "chain_matched")
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
BASE_DIR = f"{DATA_ROOT}/{NEG_RATIO}"
BASELINE_ROOT = os.path.join(
    os.environ.get("BASELINE_ROOT", f"trained_models/{MODEL_SET}"),
    f"baseline_{NEG_RATIO}")


@dataclass
class CFG:
    # paths
    meta_train_csv: str = f"{BASE_DIR}/meta_train.csv.gz"

    # the unseen test set is only used for the final report, never during training
    unseen_test_csv: str = f"{BASE_DIR}/unseen_test.csv"

    meta_test_csv: str = f"{BASE_DIR}/meta_test.csv.gz"
    few_support_csv: str = f"{BASE_DIR}/few_test_support.csv"
    few_query_csv: str = f"{BASE_DIR}/few_test_query.csv"

    baseline_fold_dir: str = f"{BASELINE_ROOT}/fold_1"
    out_root: str = os.environ.get("OUT_ROOT", f"trained_models/{MODEL_SET}/meta_{NEG_RATIO}")

    # meta-training hyperparameters
    n_folds: int = 5
    seed: int = int(os.environ.get("TRAIN_SEED", "2025"))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 5

    # outer loop
    outer_lr: float = 0.1
    outer_lr_decay: float = 0.90

    # inner loop
    inner_steps: int = 3
    inner_lr: float = 1e-3
    inner_batch_size: int = 32
    train_use_reg: bool = True

    # test-time fine-tuning (TTFT)
    ft_steps: int = 20
    ft_lr: float = 0.002
    reg_lambda: float = 0.5

    # model
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 1
    ffn_dim: int = 512
    dropout: float = 0.1
    max_cdr3: int = 60
    max_pep: int = 25
    max_mhc: int = 128
    task_col: str = "task_id"


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_csv(path):
    if not os.path.exists(path):
        alt = path[:-3] if path.endswith(".gz") else path + ".gz"
        if os.path.exists(alt):
            path = alt
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


def train_on_task_support(model, df_support, steps, lr, device, use_reg=False, strict_mode=False):
    """Inner-loop / TTFT adaptation with optional L2 anchoring to the initial parameters."""
    fmodel = copy.deepcopy(model)
    fmodel.eval()

    anchor_params = {}
    if use_reg:
        for name, param in fmodel.named_parameters():
            if param.requires_grad:
                anchor_params[name] = param.clone().detach()

    params_to_opt = [p for p in fmodel.parameters() if p.requires_grad]
    if not params_to_opt:
        params_to_opt = fmodel.parameters()
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    if strict_mode:
        # full-batch GD in numerically equivalent form: sequential micro-batches
        # with sample-count-weighted gradient accumulation, to avoid OOM when a
        # single task support set is large
        bs = min(len(df_support), 1024)
        shuffle = False
    else:
        bs = min(len(df_support), CFG.inner_batch_size)
        if bs < 2:
            bs = 2
        shuffle = True

    loader = build_loader_from_df(df_support, bs, shuffle=shuffle)
    iterator = iter(loader)
    n_total = len(df_support)

    for _ in range(steps):
        if strict_mode:
            optimizer.zero_grad()
            for batch in loader:
                for k in ["cdr3", "pep", "mhc", "y"]:
                    batch[k] = batch[k].to(device)
                logit, _ = fmodel(batch)
                bce_loss = loss_fn(logit, batch["y"]) * (batch["y"].shape[0] / n_total)
                bce_loss.backward()
            if use_reg:
                l2_loss = 0.0
                for name, param in fmodel.named_parameters():
                    if param.requires_grad:
                        l2_loss += torch.norm(param - anchor_params[name]) ** 2
                (CFG.reg_lambda * l2_loss).backward()
            optimizer.step()
            continue

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


def eval_general(model, df, device):
    model.eval()
    loader = build_loader_from_df(df, 256, shuffle=False)
    return evaluate(model, loader, device, return_arrays=False)

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

    if not all_y_true:
        return {}
    return _compute_metrics_from_arrays(np.concatenate(all_y_true), np.concatenate(all_y_prob))


def run_hybrid_fold(fold_id, train_df_fold, val_df_fold, device):
    """Train one fold: warm start from the baseline fold checkpoint, Reptile
    outer loop over tasks, model selection on the meta-train hold-out fold."""
    print(f"\n>>> Starting hybrid meta-training for fold {fold_id} <<<")

    model = AttBaselineModel(
        d_model=CFG.d_model, nhead=CFG.nhead, num_layers=CFG.num_layers,
        ffn_dim=CFG.ffn_dim, attn_dropout=CFG.dropout, emb_dropout=CFG.dropout,
        max_cdr3=CFG.max_cdr3, max_pep=CFG.max_pep, max_mhc=CFG.max_mhc
    ).to(device)

    ckpt_path = f"{CFG.baseline_fold_dir.replace('fold_1', f'fold_{fold_id}')}/best_model.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading baseline weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        print("Warning: baseline checkpoint not found, training from scratch.")

    # freeze everything, then unfreeze the prediction head, the last encoder
    # layer and encoder layer-norms
    for param in model.parameters():
        param.requires_grad = False
    trainable_params = []
    head_keywords = ["mlp", "out", "classifier", "head", "fc"]
    target_layer_idx = str(CFG.num_layers - 1)

    for name, param in model.named_parameters():
        should_train = False
        if any(k in name for k in head_keywords):
            if "enc_" not in name and "transformer" not in name:
                should_train = True
        if "layers" in name and f".{target_layer_idx}." in name:
            should_train = True
        if "norm" in name and "enc" in name:
            should_train = True

        if should_train:
            param.requires_grad = True
            trainable_params.append(name)

    task_map, task_ids = get_grouped_tasks(train_df_fold, CFG.task_col)

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    curr_outer_lr = CFG.outer_lr

    print("  Checking baseline meta-val performance...")
    base_val_res = eval_general(model, val_df_fold, device)
    print(f"  [Baseline] Meta-Val PR_AUC: {base_val_res['PR_AUC']:.4f}")

    for epoch in range(1, CFG.epochs + 1):
        np.random.shuffle(task_ids)
        theta_old_global = {k: v.clone() for k, v in model.state_dict().items()}

        for t_id in task_ids:
            df_task = task_map[t_id]

            ft_model = train_on_task_support(model, df_task,
                                             steps=CFG.inner_steps,
                                             lr=CFG.inner_lr,
                                             device=device,
                                             use_reg=CFG.train_use_reg,
                                             strict_mode=False)

            theta_new = ft_model.state_dict()
            new_state = model.state_dict()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        diff = theta_new[name] - theta_old_global[name]
                        new_state[name] = theta_old_global[name] + curr_outer_lr * diff
            model.load_state_dict(new_state)

        curr_outer_lr *= CFG.outer_lr_decay

        # model selection on the meta-train hold-out fold only
        val_metrics = eval_general(model, val_df_fold, device)
        val_pr = val_metrics["PR_AUC"]

        print(f"  [Fold {fold_id} Ep {epoch}] OuterLR={curr_outer_lr:.4f} | Meta-Val PR={val_pr:.4f}")

        if val_pr > best_score:
            best_score = val_pr
            best_state = copy.deepcopy(model.state_dict())
            print("    >>> New best model saved (based on Meta-Val).")

    print(f"Fold {fold_id} done. Best Meta-Val score: {best_score:.4f}")
    model.load_state_dict(best_state)
    return model


def main():
    set_seed(CFG.seed)
    _ensure_dir(CFG.out_root)

    print("Loading datasets...")
    df_meta_train = preprocess(load_csv(CFG.meta_train_csv))
    df_unseen_test = preprocess(load_csv(CFG.unseen_test_csv))
    df_meta_test = preprocess(load_csv(CFG.meta_test_csv))
    df_few_supp = preprocess(load_csv(CFG.few_support_csv))
    df_few_query = preprocess(load_csv(CFG.few_query_csv))

    from train_baseline import make_group_folds
    folds = make_group_folds(df_meta_train, CFG.task_col, CFG.n_folds, CFG.seed)

    results_agg = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
        df_tr = df_meta_train.iloc[tr_idx].reset_index(drop=True)
        df_va = df_meta_train.iloc[va_idx].reset_index(drop=True)

        meta_model = run_hybrid_fold(fold_id, df_tr, df_va, CFG.device)

        print(f"Fold {fold_id} final evaluation...")
        res = {"fold": fold_id}

        m_unseen = eval_general(meta_model, df_unseen_test, CFG.device)
        res["unseen_ROC"] = m_unseen["ROC_AUC"]
        res["unseen_PR"] = m_unseen["PR_AUC"]

        m_meta = eval_general(meta_model, df_meta_test, CFG.device)
        res["meta_test_ROC"] = m_meta["ROC_AUC"]
        res["meta_test_PR"] = m_meta["PR_AUC"]

        torch.save(meta_model.state_dict(), f"{CFG.out_root}/model_fold_{fold_id}.pt")
        print("  Evaluating few-shot tasks...")
        m_few = eval_few_shot_with_ttft(meta_model, df_few_supp, df_few_query,
                                        CFG.device, CFG.task_col,
                                        strict_mode=True)
        res["few_query_ROC"] = m_few.get("ROC_AUC", 0)
        res["few_query_PR"] = m_few.get("PR_AUC", 0)

        print(f"Fold {fold_id} result summary:")
        print(f"  Unseen : {res['unseen_PR']:.4f}")
        print(f"  MetaTest: {res['meta_test_PR']:.4f}")
        print(f"  FewQuery: {res['few_query_PR']:.4f}")

        results_agg.append(res)

    df_res = pd.DataFrame(results_agg)
    df_res.loc["mean"] = df_res.mean()
    df_res.to_csv(f"{CFG.out_root}/results_summary.csv")
    print("\n=== Final results summary ===")
    print(df_res)


if __name__ == "__main__":
    main()
