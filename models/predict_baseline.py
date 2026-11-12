# -*- coding: utf-8 -*-
"""
predict_baseline.py

Ensemble prediction with the 5-fold supervised baseline models.
If the input data contains labels, ROC-AUC and PR-AUC are computed and printed.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import PMHCTCRDataset, preprocess
from model import AttBaselineModel

def get_args():
    parser = argparse.ArgumentParser(description="Prediction with the 5-fold supervised baseline ensemble")

    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save prediction results")

    parser.add_argument("--model_dir", type=str, default="trained_models/chain_matched/baseline_1000x",
                        help="Directory containing fold_1..fold_5/best_model.pt")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

def load_ensemble_models(model_dir, device):
    """Load the 5 fold checkpoints from model_dir."""
    models = []
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"[Init] Loading ensemble models from: {model_dir}")

    for fold_id in range(1, 6):
        ckpt_path = os.path.join(model_dir, f"fold_{fold_id}", "best_model.pt")

        if not os.path.exists(ckpt_path):
            print(f"  [Warning] Fold {fold_id} checkpoint not found at {ckpt_path}. Skipping.")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["cfg"]  # training-time model configuration

        model = AttBaselineModel(
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            ffn_dim=cfg["ffn_dim"],
            attn_dropout=cfg["dropout"],
            emb_dropout=cfg["dropout"],
            max_cdr3=cfg["max_cdr3"],
            max_pep=cfg["max_pep"],
            max_mhc=cfg["max_mhc"],
        )

        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        models.append(model)
        print(f"  - Loaded fold {fold_id}")

    if not models:
        raise RuntimeError("No models were loaded. Check the model directory.")

    return models, cfg

def run_prediction(models, loader, device):
    """Predict with every fold model and average the probabilities."""
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            for k in ["cdr3", "pep", "mhc"]:
                batch[k] = batch[k].to(device)

            batch_preds_list = []
            for model in models:
                logit, _ = model(batch)
                prob = torch.sigmoid(logit)  # [B]
                batch_preds_list.append(prob)

            avg_prob = torch.stack(batch_preds_list).mean(dim=0)
            all_probs.extend(avg_prob.cpu().numpy().tolist())

    return all_probs

def main():
    args = get_args()

    # 1. Load data
    print(f"[Data] Reading input: {args.input_csv}")
    df_raw = pd.read_csv(args.input_csv)

    # preprocess maps TCR/MHC/label columns; missing labels are filled with 0.0
    df_proc = preprocess(df_raw)

    if len(df_proc) == 0:
        raise ValueError("Preprocessing resulted in an empty dataset. Check the TCR/MHC columns.")

    # 2. Load models
    models, cfg = load_ensemble_models(args.model_dir, args.device)

    # 3. DataLoader
    dataset = PMHCTCRDataset(
        df_proc,
        max_cdr3=cfg["max_cdr3"],
        max_pep=cfg["max_pep"],
        max_mhc=cfg["max_mhc"]
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 4. Predict
    print(f"[Pred] Start prediction on {len(df_proc)} samples...")
    probs = run_prediction(models, loader, args.device)

    # 5. Metrics (ROC-AUC, PR-AUC) when both classes are present
    y_true = df_proc["label"].values
    y_pred = np.array(probs)

    unique_labels = np.unique(y_true)
    if len(unique_labels) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)
            print("\n" + "=" * 40)
            print(f" Performance Metrics on Input Data")
            print("=" * 40)
            print(f" ROC_AUC : {roc_auc:.4f}")
            print(f" PR_AUC  : {pr_auc:.4f}")
            print("=" * 40 + "\n")
        except Exception as e:
            print(f"\n[Metrics] Error calculating metrics: {e}")
    else:
        print("\n[Metrics] Skipped: input data does not contain both positive and negative labels.")

    # 6. Save results
    df_proc["pred_prob"] = probs

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_proc.to_csv(args.output_csv, index=False)
    print(f"[Done] Results saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
