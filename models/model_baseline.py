# -*- coding: utf-8 -*-
"""
model_baseline.py
baseline: 仅使用 attention（无 9-mer register head）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, brier_score_loss
)

from dataset import VOCAB_SIZE

# ---------- Seed ----------
def set_seed(s=2025):
    import random
    import numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Blocks ----------
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len:int, d_model:int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, L = x.size()
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        return self.pe(pos)

class AttnEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, max_len=128, d_model=256, nhead=8, num_layers=1, ffn_dim=512,
                 attn_dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        if self.emb.padding_idx is not None:
            with torch.no_grad():
                self.emb.weight[self.emb.padding_idx].zero_()

        self.pos = PositionalEmbedding(max_len=max_len, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=attn_dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=max(num_layers, 1))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.attn_w = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids):
        mask = ids.eq(0)
        x = self.emb(ids) + self.pos(ids)
        x = self.emb_dropout(x)
        x = self.enc(x, src_key_padding_mask=mask)
        logits = self.attn_w(x).squeeze(-1)
        logits = logits.masked_fill(mask, float("-inf"))
        attn = torch.softmax(logits, dim=1)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)
        return self.norm(pooled), x, mask

class MaskedAttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        """
        x: [B,L,D]
        mask: [B,L]  True=pad
        """
        logit = self.w(x).squeeze(-1)              # [B,L]
        logit = logit.masked_fill(mask, float('-inf'))
        attn = torch.softmax(logit, dim=1)         # [B,L]
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # [B,D]
        return self.norm(pooled), attn

# ---------- Baseline Model ----------
class AttBaselineModel(nn.Module):
    """
    Baseline: TCR / pep / MHC 各自 self-attention 编码 + attention pooling
    无 9-mer register head
    """
    def __init__(self, d_model=256, nhead=8, num_layers=1, ffn_dim=512,
                 attn_dropout=0.1, emb_dropout=0.1, max_cdr3=60, max_pep=25, max_mhc=128):
        super().__init__()
        vocab_size = VOCAB_SIZE

        # TCR encoder（和你的主模型一致）
        self.enc_tcr = AttnEncoder(
            vocab_size=vocab_size, max_len=max_cdr3, d_model=d_model,
            nhead=nhead, num_layers=max(num_layers, 1), ffn_dim=ffn_dim,
            attn_dropout=attn_dropout, emb_dropout=emb_dropout
        )

        # Pep/MHC embedding + pos
        self.tok_pep = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tok_mhc = nn.Embedding(vocab_size, d_model, padding_idx=0)
        for emb in (self.tok_pep, self.tok_mhc):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            if emb.padding_idx is not None:
                with torch.no_grad():
                    emb.weight[emb.padding_idx].zero_()

        self.pos_pep = PositionalEmbedding(max_len=max_pep, d_model=d_model)
        self.pos_mhc = PositionalEmbedding(max_len=max_mhc, d_model=d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ffn_dim, attn_dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.enc_pep = nn.TransformerEncoder(enc_layer, num_layers=max(num_layers, 1))
        self.enc_mhc = nn.TransformerEncoder(enc_layer, num_layers=max(num_layers, 1))

        # attention pooling（pep、mhc）
        self.pep_pool = MaskedAttnPool(d_model)
        self.mhc_pool = MaskedAttnPool(d_model)

        # classifier（保持输出头形式一致）
        self.out_mlp = nn.Sequential(
            nn.Linear(3*d_model, 2*d_model), nn.GELU(), nn.Dropout(attn_dropout),
            nn.Linear(2*d_model, d_model),   nn.GELU(), nn.Dropout(attn_dropout),
            nn.Linear(d_model, 1)
        )

    def _encode_pep_mhc(self, pep_ids, mhc_ids):
        pep_mask = pep_ids.eq(0)
        mhc_mask = mhc_ids.eq(0)

        pep_x = self.tok_pep(pep_ids) + self.pos_pep(pep_ids)
        mhc_x = self.tok_mhc(mhc_ids) + self.pos_mhc(mhc_ids)

        pep_x = self.enc_pep(pep_x, src_key_padding_mask=pep_mask)
        mhc_x = self.enc_mhc(mhc_x, src_key_padding_mask=mhc_mask)
        return pep_x, pep_mask, mhc_x, mhc_mask

    def forward(self, b):
        # TCR
        tcr_vec, tcr_ctx, tcr_mask = self.enc_tcr(b["cdr3"])  # tcr_vec: [B,D]

        # Pep/MHC
        pep_x, pep_mask, mhc_x, mhc_mask = self._encode_pep_mhc(b["pep"], b["mhc"])

        # pooled vectors
        pep_vec, pep_attn = self.pep_pool(pep_x, pep_mask)    # [B,D], [B,Lp]
        mhc_vec, mhc_attn = self.mhc_pool(mhc_x, mhc_mask)    # [B,D], [B,Lm]

        # classifier
        z = torch.cat([tcr_vec, pep_vec, mhc_vec], dim=-1)
        logit = self.out_mlp(z).squeeze(-1)  # [B]

        aux = {
            "pep_attn": pep_attn,   # [B,Lp]
            "mhc_attn": mhc_attn,   # [B,Lm]
        }
        return logit, aux

# ---------- Metrics ----------
def _compute_metrics_from_arrays(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    metrics = {}
    try:
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        thr_best = thr[np.argmax(j)] if len(thr) > 0 else 0.5
    except Exception:
        thr_best = 0.5

    y_pred = (y_prob >= thr_best).astype(int)

    uniq = np.unique(y_true)
    pos = int(y_true.sum()); neg = int((1 - y_true).sum())
    metrics["N"] = int(len(y_true))
    metrics["Pos"] = pos
    metrics["Neg"] = neg
    metrics["BestThr"] = float(thr_best)
    metrics["Brier"] = brier_score_loss(y_true, y_prob)

    if len(uniq) == 2:
        metrics.update({
            "ROC_AUC": roc_auc_score(y_true, y_prob),
            "PR_AUC": average_precision_score(y_true, y_prob),
            "Acc": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "Prec": precision_score(y_true, y_pred, zero_division=0),
            "Rec": recall_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred)
        })
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["Spe"] = tn / (tn + fp + 1e-9)
    else:
        metrics.update({
            "ROC_AUC": np.nan, "PR_AUC": np.nan, "Acc": np.nan, "F1": np.nan,
            "Prec": np.nan, "Rec": np.nan, "MCC": np.nan, "Spe": np.nan
        })
    return metrics

@torch.no_grad()
def evaluate(model, loader, device, return_arrays=False):
    model.eval()
    y_true, y_prob = [], []
    for b in loader:
        for k in ["cdr3", "pep", "mhc", "y"]:
            b[k] = b[k].to(device)
        logit, _ = model(b)
        p = torch.sigmoid(logit).cpu().numpy()
        y = b["y"].cpu().numpy()
        y_true.append(y); y_prob.append(p)
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    metrics = _compute_metrics_from_arrays(y_true, y_prob)
    if return_arrays:
        return metrics, y_true, y_prob
    return metrics
