# -*- coding: utf-8 -*-
"""
plot_epitope_attention_5fold_9mer_large_font.py

功能：
1. 核心区域定义为连续 9-mer。
2. 可视化美化：Arial 字体，配色柔和。
3. **特大字体优化**：适用于海报展示或缩小插入论文时的清晰度。

输出：
- Residue Heatmap
- 9-mer Window Heatmap
- Mean Residue Bar (+ Core highlight)
- Mean 9-mer Curve (+ Peak highlight)
- Summary CSV

用法示例：
python plot_epitope_attention_5fold_9mer_large_font.py \
  --input data.csv \
  --model_dir result/exp2_healthy_1000x_5fold \
  --out_dir attn_out_large_font \
  --top_n 50 \
  --sort_by prob \
  --device cuda
"""

import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl

from dataset import PMHCTCRDataset, preprocess
from model_baseline import AttBaselineModel


# =========================
# Global Plotting Styles (Large Fonts & Arial)
# =========================
# 1. 设置字体
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False

# 2. 设置超大字体参数 (Poster / High-Vis Style)
base_size = 18  # 基础字号
mpl.rcParams['font.size'] = base_size
mpl.rcParams['axes.titlesize'] = base_size + 6  # 标题 24
mpl.rcParams['axes.labelsize'] = base_size + 2  # 轴标签 20
mpl.rcParams['xtick.labelsize'] = base_size      # 刻度 18
mpl.rcParams['ytick.labelsize'] = base_size      # 刻度 18
mpl.rcParams['legend.fontsize'] = base_size - 2  # 图例 16
mpl.rcParams['figure.titlesize'] = base_size + 8

# 3. 布局与线条
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['lines.linewidth'] = 2.5  # 线条加粗
mpl.rcParams['patch.linewidth'] = 1.5  # 边框加粗

# 4. 自定义颜色
COLOR_BAR = '#4e79a7'       # 蓝色
COLOR_HIGHLIGHT = '#e15759' # 红色
COLOR_LINE = '#f28e2b'      # 橙色


# =========================
# Config
# =========================
@dataclass
class CFG:
    n_folds: int = 5
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 1
    ffn_dim: int = 512
    dropout: float = 0.1
    max_cdr3: int = 60
    max_pep: int = 25
    max_mhc: int = 128


# =========================
# Utils
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_ensemble_models(root_dir: str, device: str, cfg: CFG) -> List[torch.nn.Module]:
    models = []
    print(f"[Load] Loading {cfg.n_folds}-fold ensemble from: {root_dir}")
    for fold in range(1, cfg.n_folds + 1):
        path_v1 = os.path.join(root_dir, f"fold_{fold}", "best_model.pt")
        path_v2 = os.path.join(root_dir, f"model_fold_{fold}.pt")
        if os.path.exists(path_v1):
            path = path_v1
        elif os.path.exists(path_v2):
            path = path_v2
        else:
            print(f"  [Warn] Fold {fold} not found. Skipping.")
            continue

        model = AttBaselineModel(
            d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.num_layers,
            ffn_dim=cfg.ffn_dim, attn_dropout=cfg.dropout, emb_dropout=cfg.dropout,
            max_cdr3=cfg.max_cdr3, max_pep=cfg.max_pep, max_mhc=cfg.max_mhc
        ).to(device)
        ckpt = torch.load(path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
        print(f"  [OK] fold {fold}: {path}")

    if len(models) == 0:
        raise RuntimeError("No models loaded.")
    return models


@torch.no_grad()
def infer_probs_and_pep_attn(
    models: List[torch.nn.Module],
    df: pd.DataFrame,
    device: str,
    cfg: CFG,
    batch_size: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ds = PMHCTCRDataset(df, max_cdr3=cfg.max_cdr3, max_pep=cfg.max_pep, max_mhc=cfg.max_mhc)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    K = len(models)
    probs_all = [[] for _ in range(K)]
    attn_all = [[] for _ in range(K)]
    epis: List[str] = []

    for b in loader:
        for k in ["cdr3", "pep", "mhc"]:
            b[k] = b[k].to(device)
        epis.extend(list(b["epi"]))
        for mi, model in enumerate(models):
            logit, aux = model(b)
            prob = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)
            pep_attn = aux["pep_attn"].detach().cpu().numpy()
            probs_all[mi].append(prob)
            attn_all[mi].append(pep_attn)

    probs_k = np.stack([np.concatenate(probs_all[i], axis=0) for i in range(K)], axis=0)
    attn_k = np.stack([np.concatenate(attn_all[i], axis=0) for i in range(K)], axis=0)
    return probs_k, attn_k, epis


def normalize_rows(A: np.ndarray) -> np.ndarray:
    return A / (A.sum(axis=1, keepdims=True) + 1e-12)


def compute_9mer_scores_from_res_attn(attn_1d: np.ndarray, window: int = 9) -> np.ndarray:
    L = len(attn_1d)
    if L < window:
        return np.array([], dtype=float)
    kernel = np.ones(window)
    scores = np.convolve(attn_1d, kernel, mode='valid')
    return scores


def compute_9mer_scores_folds(attn_folds_L: np.ndarray, window: int = 9) -> np.ndarray:
    K, L = attn_folds_L.shape
    if L < window:
        return np.zeros((K, 0), dtype=float)
    out = []
    for k in range(K):
        out.append(compute_9mer_scores_from_res_attn(attn_folds_L[k], window=window))
    return np.stack(out, axis=0)


def top_k_9mers(mean_9mer: np.ndarray, epitope: str, k: int = 3, window: int = 9):
    if mean_9mer.size == 0:
        return [], [], []
    idx = np.argsort(-mean_9mer)[: min(k, mean_9mer.size)]
    starts_1based = [int(i) + 1 for i in idx]
    seqs = [epitope[i:i+window] for i in idx]
    scores = [float(mean_9mer[i]) for i in idx]
    # Sort by position
    order = np.argsort(starts_1based)
    starts_1based = [starts_1based[j] for j in order]
    seqs = [seqs[j] for j in order]
    scores = [scores[j] for j in order]
    return starts_1based, seqs, scores


# =========================
# Plotting Functions (Large Fonts)
# =========================
def plot_residue_heatmap(epi: str, attn_folds: np.ndarray, title: str, savepath: str):
    L = len(epi)
    A = attn_folds[:, :L].copy()
    A = normalize_rows(A)

    width = max(10, L * 0.6)
    height = 5.0
    
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(A, aspect="auto", cmap="viridis", vmin=0, vmax=A.max())

    cbar = plt.colorbar(im, fraction=0.03, pad=0.02, ax=ax)
    cbar.set_label('Attention Weight', fontsize=16)

    ax.set_yticks(range(A.shape[0]))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(A.shape[0])])
    ax.set_xticks(range(L))
    ax.set_xticklabels(list(epi), fontfamily='monospace', fontweight='bold')

    ax.set_xlabel("Epitope Sequence (AA Position)")
    ax.set_ylabel("Model Fold")
    ax.set_title(title, fontweight='bold', pad=15)

    ax.grid(False)
    plt.tight_layout()
    # 同时保存 PNG 和 PDF
    plt.savefig(savepath)
    plt.savefig(savepath.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.close()


def plot_9mer_heatmap(epi: str, attn_folds: np.ndarray, title: str, savepath: str, window: int = 9):
    L = len(epi)
    A = attn_folds[:, :L].copy()
    A = normalize_rows(A)
    nine = compute_9mer_scores_folds(A, window=window)

    if nine.shape[1] == 0:
        return

    W = nine.shape[1]
    width = max(10, W * 0.7) 
    height = 5.0

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(nine, aspect="auto", cmap="magma", vmin=0, vmax=nine.max())

    cbar = plt.colorbar(im, fraction=0.03, pad=0.02, ax=ax)
    cbar.set_label('9-mer Summed Attention', fontsize=16)

    ax.set_yticks(range(nine.shape[0]))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(nine.shape[0])])

    xtick_labels = [str(i) for i in range(1, W + 1)]
    ax.set_xticks(range(W))
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel("9-mer Start Position (1-based)")
    ax.set_ylabel("Model Fold")
    ax.set_title(title, fontweight='bold', pad=15)

    ax.grid(False)
    plt.tight_layout()
    # 同时保存 PNG 和 PDF
    plt.savefig(savepath)
    plt.savefig(savepath.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.close()


def plot_mean_residue_with_9mer_overlay(epi: str, attn_folds: np.ndarray, title: str, savepath: str, window: int = 9):
    L = len(epi)
    A = attn_folds[:, :L].copy()
    A = normalize_rows(A)
    mean_attn = A.mean(axis=0)
    mean_attn = mean_attn / (mean_attn.sum() + 1e-12)

    nine = compute_9mer_scores_from_res_attn(mean_attn, window=window)
    best_i_0based = int(nine.argmax()) if nine.size > 0 else None

    x = np.arange(1, L + 1)
    width = max(10, L * 0.6)
    height = 5.5

    fig, ax = plt.subplots(figsize=(width, height))
    bars = ax.bar(x, mean_attn, color=COLOR_BAR, alpha=0.8, edgecolor='grey', linewidth=1.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(list(epi), fontfamily='monospace', fontweight='bold')
    ax.set_xlabel("Residue Position & Sequence")
    ax.set_ylabel("Mean Attention")
    ax.set_title(title, fontweight='bold', pad=15)

    if best_i_0based is not None:
        start_1b = best_i_0based + 1
        end_1b = best_i_0based + window
        ax.axvspan(start_1b - 0.5, end_1b + 0.5, color=COLOR_HIGHLIGHT, alpha=0.25, zorder=2, label='Top-1 Core')

        core_seq = epi[best_i_0based : best_i_0based + window]
        text_str = f"Core ({start_1b}-{end_1b}): {core_seq}"
        props = dict(boxstyle='round', facecolor=COLOR_HIGHLIGHT, alpha=0.15, edgecolor=COLOR_HIGHLIGHT)
        ax.text(0.02, 0.94, text_str, transform=ax.transAxes, fontsize=18, fontweight='bold',
                verticalalignment='top', bbox=props)
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    ax.set_ylim(bottom=0)
    plt.tight_layout()
    # 同时保存 PNG 和 PDF
    plt.savefig(savepath)
    plt.savefig(savepath.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.close()


def plot_mean_9mer_curve(epi: str, attn_folds: np.ndarray, title: str, savepath: str, window: int = 9):
    L = len(epi)
    A = attn_folds[:, :L].copy()
    A = normalize_rows(A)
    mean_attn = A.mean(axis=0)
    mean_attn = mean_attn / (mean_attn.sum() + 1e-12)

    nine = compute_9mer_scores_from_res_attn(mean_attn, window=window)
    if nine.size == 0: return

    best_i_0based = int(nine.argmax())
    W = nine.size
    x = np.arange(1, W + 1)
    width = max(10, W * 0.7)
    height = 5.5

    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(x, nine, marker="o", color=COLOR_LINE, linewidth=3, markersize=10, alpha=0.8, zorder=3)
    ax.scatter([best_i_0based + 1], [nine[best_i_0based]], color=COLOR_HIGHLIGHT, s=200, marker='*', zorder=4, label="Max Score")

    ax.set_xticks(x)
    ax.set_xlabel("9-mer Start Position")
    ax.set_ylabel("Mean 9-mer Score")
    ax.set_title(title, fontweight='bold', pad=15)

    core_seq = epi[best_i_0based : best_i_0based + window]
    text_str = f"Pos: {best_i_0based+1}\nSeq: {core_seq}\nScore: {nine[best_i_0based]:.3f}"
    props = dict(boxstyle='round', facecolor=COLOR_LINE, alpha=0.15, edgecolor=COLOR_LINE)
    x_txt_pos = 0.75 if best_i_0based < W / 2 else 0.05
    ax.text(x_txt_pos, 0.94, text_str, transform=ax.transAxes, fontsize=18, fontweight='bold',
            verticalalignment='top', bbox=props)
    
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    plt.tight_layout()
    # 同时保存 PNG 和 PDF
    plt.savefig(savepath)
    plt.savefig(savepath.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.close()


# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV")
    p.add_argument("--model_dir", required=True, help="Directory containing 5-fold checkpoints")
    p.add_argument("--out_dir", default="attn_out_large", help="Output directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--top_n", type=int, default=50, help="How many samples to plot")
    p.add_argument("--sort_by", choices=["prob", "none"], default="prob")
    p.add_argument("--min_len", type=int, default=1)
    p.add_argument("--max_len", type=int, default=25)
    p.add_argument("--window", type=int, default=9, help="Core window length")
    p.add_argument("--top_k_9mer", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG()

    if args.max_len > cfg.max_pep:
        raise ValueError(f"--max_len ({args.max_len}) cannot exceed model max_pep ({cfg.max_pep}).")

    ensure_dir(args.out_dir)
    plot_dir = os.path.join(args.out_dir, "plots")
    ensure_dir(plot_dir)

    # 1) Load Data
    df_raw = pd.read_csv(args.input)
    df = preprocess(df_raw)
    lens = df["Epitope.peptide"].astype(str).str.len()
    df = df.loc[(lens >= args.min_len) & (lens <= args.max_len)].reset_index(drop=True)
    
    if len(df) == 0:
        raise RuntimeError("No samples left after length filtering.")

    # 2) Load Models
    models = load_ensemble_models(args.model_dir, args.device, cfg)

    # 3) Inference
    probs_k, attn_k, epis = infer_probs_and_pep_attn(models, df, args.device, cfg, args.batch_size)
    avg_prob = probs_k.mean(axis=0)

    # 4) Select samples
    N = len(epis)
    if args.sort_by == "prob":
        order = np.argsort(-avg_prob)
    else:
        order = np.arange(N)
    order = order[: min(args.top_n, N)]

    # 5) Plot & Summary
    summary_rows = []
    print(f"[Plotting] Generating LARGE FONT plots for top {len(order)} samples...")

    for rank, i in enumerate(order, 1):
        epi = epis[i]
        L = len(epi)
        fold_attn = attn_k[:, i, :] 
        fold_prob = probs_k[:, i]

        safe_epi = "".join([c if c.isalnum() else "_" for c in epi])
        base = f"rank{rank:03d}_idx{i:06d}_prob{avg_prob[i]:.4f}_{safe_epi}"

        # --- Plots ---
        res_hm_path = os.path.join(plot_dir, base + "_1_residue_hm.png")
        plot_residue_heatmap(
            epi=epi, attn_folds=fold_attn,
            title=f"Residue Attention (AvgProb: {avg_prob[i]:.4f})",
            savepath=res_hm_path
        )

        res_bar_path = os.path.join(plot_dir, base + "_2_mean_residue_9mer.png")
        plot_mean_residue_with_9mer_overlay(
            epi=epi, attn_folds=fold_attn,
            title=f"Mean Attention & Top Core (W={args.window})",
            savepath=res_bar_path, window=args.window
        )

        nine_hm_path = ""
        nine_curve_path = ""
        best_9mer_data = {}

        # 9-mer logic
        A_mean = normalize_rows(fold_attn[:, :L]).mean(axis=0)
        A_mean = A_mean / (A_mean.sum() + 1e-12)
        mean_9mer = compute_9mer_scores_from_res_attn(A_mean, window=args.window)

        if mean_9mer.size > 0:
            best_i = int(mean_9mer.argmax())
            best_9mer_data = {
                "best_9mer_start_1b": best_i + 1,
                "best_9mer_seq": epi[best_i:best_i + args.window],
                "best_9mer_score": mean_9mer[best_i]
            }
            starts, seqs, scores = top_k_9mers(mean_9mer, epi, k=args.top_k_9mer, window=args.window)
            best_9mer_data["topk_starts"] = ";".join(map(str, starts))
            best_9mer_data["topk_seqs"] = ";".join(seqs)
            best_9mer_data["topk_scores"] = ";".join([f"{s:.4f}" for s in scores])

            nine_hm_path = os.path.join(plot_dir, base + "_3_9mer_hm.png")
            plot_9mer_heatmap(
                epi=epi, attn_folds=fold_attn,
                title=f"9-mer Window Attention",
                savepath=nine_hm_path, window=args.window
            )

            nine_curve_path = os.path.join(plot_dir, base + "_4_mean_9mer_curve.png")
            plot_mean_9mer_curve(
                epi=epi, attn_folds=fold_attn,
                title=f"Mean 9-mer Attention Curve",
                savepath=nine_curve_path, window=args.window
            )

        row = {
            "rank": rank,
            "original_index": i,
            "epitope": epi,
            "length": L,
            "avg_prob": float(avg_prob[i]),
            "fold_probs": ";".join([f"{p:.4f}" for p in fold_prob]),
            "best_9mer_seq": best_9mer_data.get("best_9mer_seq", ""),
            "plot_residue_hm": res_hm_path,
        }
        summary_rows.append(row)

    out_csv = os.path.join(args.out_dir, f"summary_large_font.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"[Done] Large font plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()