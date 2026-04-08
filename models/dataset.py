# -*- coding: utf-8 -*-
"""
dataset.py
包含数据集、预处理、MHC伪序列加载、分词等
"""

import os, sys, json, random, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
PSEUDO_DIR = ROOT / "mhc_pseudoseq"

# ---------- Tokenization ----------
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
AA_TO_ID = {a: i+1 for i,a in enumerate(AA_VOCAB)}  # 0 预留给 PAD
VOCAB_SIZE = len(AA_VOCAB) + 1  # 含 PAD

def clean_seq(seq:str)->str:
    if not isinstance(seq,str): return ""
    s = seq.strip().upper().replace(" ","")
    return "".join([a if a in AA_TO_ID else "X" for a in s])

def tokenize(seq,max_len):
    s = clean_seq(seq)
    ids = [AA_TO_ID.get(a,AA_TO_ID["X"]) for a in s[:max_len]]
    if len(ids)<max_len: ids += [0]*(max_len-len(ids))
    return np.array(ids,dtype=np.int64)

# ---------- Inline MHC pseudo loader (processed_seq_1) ----------
FILE_MAP = {
    "DPA1": "DPA1_seq.csv",
    "DPB1": "DPB1_seq.csv",
    "DQA1": "DQA1_seq.csv",
    "DQA2": "DQA2_seq.csv",
    "DQB1": "DQB1_seq.csv",
    "DQB2": "DQB2_seq.csv",
    "DRA":  "DRA_seq.csv",
    "DRB":  "DRB_seq.csv",  # 包含 DRB1/3/4/5
}
ALLELE_HEADS = (
    "DRA","DRB1","DRB3","DRB4","DRB5",
    "DQA1","DQA2","DQB1","DQB2",
    "DPA1","DPB1",
)
_PSEUDO_INDEX = {}

def _make_aliases(prot_raw: str):
    if not isinstance(prot_raw, str) or not prot_raw:
        return set()
    p = prot_raw.strip().upper().replace(" ", "")
    aliases = set([p])
    m = re.match(r'^(%s)(?:\*|_|)?(\d{2})(?::?(\d{2}))?$' % "|".join(ALLELE_HEADS), p)
    if m:
        head, a1, a2 = m.group(1), m.group(2), m.group(3)
    else:
        m2 = re.match(r'^(%s)(\d{4})$' % "|".join(ALLELE_HEADS), p)
        if m2:
            head = m2.group(1)
            a1, a2 = m2.group(2)[:2], m2.group(2)[2:]
        else:
            m3 = re.match(r'^(%s)\*(\d{4})$' % "|".join(ALLELE_HEADS), p)
            if m3:
                head = m3.group(1)
                a1, a2 = m3.group(2)[:2], m3.group(2)[2:]
            else:
                return aliases
    head = head.upper()
    if a2 is None:
        aliases.update([f"{head}*{a1}", f"{head}_{a1}", f"{head}{a1}"])
    else:
        aliases.update([f"{head}*{a1}:{a2}", f"{head}*{a1}{a2}", f"{head}_{a1}{a2}", f"{head}{a1}{a2}"])
    return aliases

def _gene_key_from_head(head: str) -> str:
    head = head.upper()
    if head.startswith("DRB"):
        return "DRB"
    return head

def _parse_allele_to_aliases(allele: str):
    if not isinstance(allele, str): 
        return None, set()
    s = allele.strip().upper().replace(" ", "")
    if s == "" or s == "NAN":
        return None, set()
    m = re.match(r'^(%s)' % "|".join(ALLELE_HEADS), s)
    if not m:
        return None, set()
    head = m.group(1)
    gene_key = _gene_key_from_head(head)
    aliases = _make_aliases(s)
    suf = s[len(head):]
    suf = suf.lstrip("*_")
    m2 = re.match(r'^(\d{2})(?::?(\d{2}))?$', suf) or re.match(r'^(\d{4})$', suf)
    if m2:
        if len(m2.groups()) == 2 and m2.group(2) is not None:
            a1, a2 = m2.group(1), m2.group(2)
        else:
            digits = m2.group(1)
            if len(digits) >= 4:
                a1, a2 = digits[:2], digits[2:4]
            else:
                a1, a2 = digits[:2], None
        if a2 is None:
            aliases.update({f"{head}*{a1}", f"{head}_{a1}", f"{head}{a1}"})
        else:
            aliases.update({f"{head}*{a1}:{a2}", f"{head}*{a1}{a2}", f"{head}_{a1}{a2}", f"{head}{a1}{a2}"})
    return gene_key, aliases

def _build_pseudo_index():
    index = {}
    for gk, fname in FILE_MAP.items():
        fpath = PSEUDO_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"[MHC pseudo] 文件缺失：{fpath}")
        df = pd.read_csv(fpath, dtype=str)
        if "prot" not in df.columns or "processed_seq_1" not in df.columns:
            raise ValueError(f"[MHC pseudo] {fpath} 缺少 'prot' 或 'processed_seq_1' 列")
        mapping = {}
        for _, row in df.iterrows():
            prot = str(row["prot"]).strip()
            seq  = str(row["processed_seq_1"]).strip()
            if not prot or not seq:
                continue
            aliases = _make_aliases(prot)
            aliases.add(prot.upper())
            for a in aliases:
                mapping[a] = seq
        index[gk] = mapping
    return index

def get_pseudoseq_inline(allele: str) -> str:
    global _PSEUDO_INDEX
    if not _PSEUDO_INDEX:
        _PSEUDO_INDEX = _build_pseudo_index()
    gene_key, aliases = _parse_allele_to_aliases(allele)
    if gene_key is None or not aliases:
        return ""
    mp = _PSEUDO_INDEX.get(gene_key, {})
    # 直接检查别名映射
    for alias in aliases:
        if alias in mp:
            return mp[alias]
    return ""

# ---------- Dataset ----------
class PMHCTCRDataset(Dataset):
    def __init__(self,df,max_cdr3=60,max_pep=25,max_mhc=128):
        self.df=df.reset_index(drop=True)
        self.max_cdr3=max_cdr3;self.max_pep=max_pep;self.max_mhc=max_mhc
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]
        return {
            "cdr3":torch.from_numpy(tokenize(r["CDR3.beta.aa"],self.max_cdr3)),
            "pep": torch.from_numpy(tokenize(r["Epitope.peptide"],self.max_pep)),
            "mhc": torch.from_numpy(tokenize(r["MHC_pseudo"],self.max_mhc)),
            "y":   torch.tensor(np.float32(r["label"])),
            "epi": r.get("Epitope.peptide","")  # for SupCon
        }

# ---------- Preprocess ----------
def preprocess(df):
    df = df.copy()

    # ---- TCR 合并 ----
    tra = df.get("cdr3_TRA", pd.Series([""] * len(df))).fillna("").astype(str)
    trb = df.get("cdr3_TRB", pd.Series([""] * len(df))).fillna("").astype(str)
    df["CDR3.beta.aa"] = [(a + b if a and b else a or b or "") for a, b in zip(tra, trb)]
    n_before = len(df)
    df = df[df["CDR3.beta.aa"] != ""]
    n_drop_tcr = n_before - len(df)

    # ---- 选择 MHC 列：优先 norm ----
    if "MHCA_norm" in df.columns and "MHCB_norm" in df.columns:
        mhca = df["MHCA_norm"].fillna("").astype(str)
        mhcb = df["MHCB_norm"].fillna("").astype(str)
    else:
        mhca = df.get("MHCA", pd.Series([""] * len(df))).fillna("").astype(str)
        mhcb = df.get("MHCB", pd.Series([""] * len(df))).fillna("").astype(str)

    a_pseudo = mhca.apply(get_pseudoseq_inline)
    b_pseudo = mhcb.apply(get_pseudoseq_inline)
    df["MHC_pseudo"] = [(ap + bp if ap and bp else ap or bp or "") for ap, bp in zip(a_pseudo, b_pseudo)]

    n_before = len(df)
    df = df[df["MHC_pseudo"] != ""]
    n_drop_mhc = n_before - len(df)

    # ---- epitope ----
    df["Epitope.peptide"] = df.get("epitope", pd.Series([""] * len(df))).fillna("").astype(str)

    # ---- label：优先 Target，其次 label，如果都没有则填 0.0 (支持无标签预测) ----
    if "Target" in df.columns:
        # 兼容部分数据集使用 Target 作为标签名
        df["label"] = pd.to_numeric(df["Target"], errors="coerce").fillna(0).astype(float)
    elif "label" in df.columns:
        # 标准情况
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(float)
    else:
        # 【关键修改】：如果没有标签列，直接创建全 0 列，避免报错
        df["label"] = 0.0

    print(f"[Preprocess] dropped TCR-empty: {n_drop_tcr}, dropped MHC-empty: {n_drop_mhc}, kept: {len(df)}")
    return df.reset_index(drop=True)