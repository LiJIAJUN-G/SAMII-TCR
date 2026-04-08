# -*- coding: utf-8 -*-
"""
predict_baseline.py
加载训练好的 Baseline (5-fold) 模型，对新数据进行集成预测。
如果输入数据包含标签，会自动计算并打印 ROC_AUC 和 PR_AUC。
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

# 导入您提供的模块
from dataset import PMHCTCRDataset, preprocess
from model_baseline import AttBaselineModel

def get_args():
    parser = argparse.ArgumentParser(description="Predict utilizing 5-fold Baseline Ensemble")
    
    # 输入输出
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save prediction results")
    
    # 模型选择
    parser.add_argument("--variant", type=str, default="exp2_healthy_100x", 
                        choices=["exp2_healthy_10x", "exp2_healthy_100x", "exp2_healthy_1000x"],
                        help="Which training variant to use (folder name suffix)")
    
    # 路径配置 (需要与 train_exp2_baseline.py 中的 out_root 对应)
    parser.add_argument("--model_root", type=str, default="result/exp2_cross_5fold_baseline",
                        help="Root directory where models are saved")
    
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

def load_ensemble_models(model_root, variant, device):
    """
    加载指定变体的 5 个 fold 的模型
    """
    models = []
    # 训练脚本生成的目录名为 train_{variant}
    train_dir = os.path.join(model_root, f"train_{variant}")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Model directory not found: {train_dir}")

    print(f"[Init] Loading ensemble models from: {train_dir}")
    
    # 遍历 5 个 fold
    for fold_id in range(1, 6):
        ckpt_path = os.path.join(train_dir, f"fold_{fold_id}", "best_model.pt")
        
        if not os.path.exists(ckpt_path):
            print(f"  [Warning] Fold {fold_id} checkpoint not found at {ckpt_path}. Skipping.")
            continue
            
        # 加载 checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["cfg"] # 读取训练时的配置
        
        # 初始化模型
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
        print(f"  - Loaded Fold {fold_id}")

    if not models:
        raise RuntimeError("No models were loaded! Check your paths.")
        
    return models, cfg

def run_prediction(models, loader, device):
    """
    运行预测：对每个 batch 使用所有模型进行预测，然后取平均值
    """
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            # Move data to device
            for k in ["cdr3", "pep", "mhc"]:
                batch[k] = batch[k].to(device)
            
            # Ensemble prediction for this batch
            batch_preds_list = []
            for model in models:
                logit, _ = model(batch)
                prob = torch.sigmoid(logit) # [B]
                batch_preds_list.append(prob)
            
            # Stack and Mean: [5, B] -> [B]
            avg_prob = torch.stack(batch_preds_list).mean(dim=0)
            
            all_probs.extend(avg_prob.cpu().numpy().tolist())
            
    return all_probs

def main():
    args = get_args()
    
    # 1. 加载数据
    print(f"[Data] Reading input: {args.input_csv}")
    df_raw = pd.read_csv(args.input_csv)
    
    # 使用 dataset.py 中的预处理 (处理 TCR/MHC/Label 等)
    # 警告：dataset.py 会将缺失的标签填充为 0.0
    df_proc = preprocess(df_raw)
    
    if len(df_proc) == 0:
        raise ValueError("Preprocessing resulted in empty dataset. Check TCR/MHC columns.")

    # 2. 加载模型
    models, cfg = load_ensemble_models(args.model_root, args.variant, args.device)
    
    # 3. 构建 DataLoader
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
    
    # 4. 预测
    print(f"[Pred] Start prediction on {len(df_proc)} samples...")
    probs = run_prediction(models, loader, args.device)
    
    # 5. 计算并打印指标 (ROC_AUC, PR_AUC)
    # dataset.py 预处理保证了 'label' 列存在
    y_true = df_proc["label"].values
    y_pred = np.array(probs)
    
    # 检查是否包含至少两个类别 (0和1)，否则无法计算 AUC
    unique_labels = np.unique(y_true)
    if len(unique_labels) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)
            print("\n" + "="*40)
            print(f" Performance Metrics on Input Data")
            print("="*40)
            print(f" ROC_AUC : {roc_auc:.4f}")
            print(f" PR_AUC  : {pr_auc:.4f}")
            print("="*40 + "\n")
        except Exception as e:
            print(f"\n[Metrics] Error calculating metrics: {e}")
    else:
        print("\n[Metrics] Skipped: Input data does not contain both positive and negative labels.")

    # 6. 保存结果
    df_proc["pred_prob"] = probs
    
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    df_proc.to_csv(args.output_csv, index=False)
    print(f"[Done] Results saved to: {args.output_csv}")

if __name__ == "__main__":
    main()