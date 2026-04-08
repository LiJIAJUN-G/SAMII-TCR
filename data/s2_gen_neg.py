# -*- coding: utf-8 -*-
"""
s2_gen_neg_fixed_fast.py

优化版本：大幅提升阴性样本生成速度
"""

import os
import numpy as np
import pandas as pd

# =========================
# 配置区
# =========================
RANDOM_SEED = 2025
IN_SPLIT_DIR = "splits_meta"
OUT_DIR = "splits_meta_with_neg"
HEALTHY_TCR_CSV = "raw/healthy_hum/healthy_tcr.csv"
SPLIT_FILES = [
    "meta_train.csv", "meta_val.csv", "meta_test.csv", 
    "few_test_support.csv", "few_test_query.csv", "unseen_test.csv"
]

COL_TRA = "cdr3_TRA"
COL_TRB = "cdr3_TRB"
COL_EPI = "epitope"
COL_MHCA = "MHCA_norm"
COL_MHCB = "MHCB_norm"
COL_Y = "Target"

PMHC_NEG_MULT = 5
EXP1_TCR_MULT = 5
EXP2_HEALTHY_MULTS = [10, 100, 1000]
MAX_RESAMPLE_TRIES = 10  # 减少重试次数

# =========================
# 优化的阴性生成函数
# =========================

def get_original_pairs_fast(pos: pd.DataFrame):
    """快速获取原始阳性配对标识"""
    # 使用字符串哈希作为唯一标识，避免逐行比较
    identifiers = []
    for idx, row in pos.iterrows():
        tra = str(row[COL_TRA]) if pd.notna(row[COL_TRA]) else "NA"
        trb = str(row[COL_TRB]) if pd.notna(row[COL_TRB]) else "NA"
        epi = str(row[COL_EPI]) if pd.notna(row[COL_EPI]) else "NA"
        mhca = str(row[COL_MHCA]) if pd.notna(row[COL_MHCA]) else "NA"
        mhcb = str(row[COL_MHCB]) if pd.notna(row[COL_MHCB]) else "NA"
        identifier = f"{tra}|{trb}|{epi}|{mhca}|{mhcb}"
        identifiers.append(identifier)
    return set(identifiers)

def get_original_pmhc_triplets_fast(pos: pd.DataFrame):
    """快速获取原始pMHC三元组标识"""
    identifiers = []
    for idx, row in pos.iterrows():
        epi = str(row[COL_EPI]) if pd.notna(row[COL_EPI]) else "NA"
        mhca = str(row[COL_MHCA]) if pd.notna(row[COL_MHCA]) else "NA"
        mhcb = str(row[COL_MHCB]) if pd.notna(row[COL_MHCB]) else "NA"
        identifier = f"{epi}|{mhca}|{mhcb}"
        identifiers.append(identifier)
    return set(identifiers)

def gen_neg_pmhc_break_pair_fast(pos: pd.DataFrame, mult: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    快速版本：pmhc_not_bind阴性生成
    """
    n = len(pos)
    if n == 0 or mult <= 0:
        return pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})

    # 获取原始配对标识
    original_pairs = get_original_pairs_fast(pos)
    original_triplets = get_original_pmhc_triplets_fast(pos)
    
    pos = pos.reset_index(drop=True)
    
    neg_blocks = []
    
    for batch in range(mult):
        print(f"      Batch {batch+1}/{mult}", end="\r")
        
        # 批量shuffle
        epi_idx = rng.permutation(n)
        mhc_idx = rng.permutation(n)
        
        # 批量创建阴性样本
        neg = pos.copy()
        neg[COL_EPI] = pos[COL_EPI].iloc[epi_idx].reset_index(drop=True).values
        neg[COL_MHCA] = pos[COL_MHCA].iloc[mhc_idx].reset_index(drop=True).values
        neg[COL_MHCB] = pos[COL_MHCB].iloc[mhc_idx].reset_index(drop=True).values
        
        # 批量检查并修正重复配对
        needs_resample = np.zeros(n, dtype=bool)
        
        # 第一遍：快速标识需要重采样的样本
        for i in range(n):
            tra = str(neg.at[i, COL_TRA]) if pd.notna(neg.at[i, COL_TRA]) else "NA"
            trb = str(neg.at[i, COL_TRB]) if pd.notna(neg.at[i, COL_TRB]) else "NA"
            epi = str(neg.at[i, COL_EPI]) if pd.notna(neg.at[i, COL_EPI]) else "NA"
            mhca = str(neg.at[i, COL_MHCA]) if pd.notna(neg.at[i, COL_MHCA]) else "NA"
            mhcb = str(neg.at[i, COL_MHCB]) if pd.notna(neg.at[i, COL_MHCB]) else "NA"
            
            pair_id = f"{tra}|{trb}|{epi}|{mhca}|{mhcb}"
            triplet_id = f"{epi}|{mhca}|{mhcb}"
            
            if pair_id in original_pairs:
                needs_resample[i] = True
        
        # 第二遍：批量重采样
        if needs_resample.any():
            resample_indices = np.where(needs_resample)[0]
            for i in resample_indices:
                tries = 0
                while tries < MAX_RESAMPLE_TRIES:
                    # 随机选择新的pMHC
                    new_epi_idx = rng.choice(n)
                    new_mhc_idx = rng.choice(n)
                    
                    new_epi = pos.at[new_epi_idx, COL_EPI]
                    new_mhca = pos.at[new_mhc_idx, COL_MHCA]
                    new_mhcb = pos.at[new_mhc_idx, COL_MHCB]
                    
                    # 检查新组合是否不同
                    tra = str(neg.at[i, COL_TRA]) if pd.notna(neg.at[i, COL_TRA]) else "NA"
                    trb = str(neg.at[i, COL_TRB]) if pd.notna(neg.at[i, COL_TRB]) else "NA"
                    epi_str = str(new_epi) if pd.notna(new_epi) else "NA"
                    mhca_str = str(new_mhca) if pd.notna(new_mhca) else "NA"
                    mhcb_str = str(new_mhcb) if pd.notna(new_mhcb) else "NA"
                    
                    new_pair_id = f"{tra}|{trb}|{epi_str}|{mhca_str}|{mhcb_str}"
                    
                    if new_pair_id not in original_pairs:
                        neg.at[i, COL_EPI] = new_epi
                        neg.at[i, COL_MHCA] = new_mhca
                        neg.at[i, COL_MHCB] = new_mhcb
                        break
                    tries += 1
        
        neg[COL_Y] = 0
        neg_blocks.append(neg)
    
    print()  # 换行
    if neg_blocks:
        neg_df = pd.concat(neg_blocks, axis=0, ignore_index=True)
    else:
        neg_df = pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})
    
    return neg_df

def gen_neg_tcr_shuffle_fast(pos: pd.DataFrame, mult: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    快速版本：tcr_shuffle阴性生成
    """
    n = len(pos)
    if n == 0 or mult <= 0:
        return pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})

    original_pairs = get_original_pairs_fast(pos)
    pos = pos.reset_index(drop=True)
    
    neg_blocks = []
    
    for batch in range(mult):
        print(f"      Batch {batch+1}/{mult}", end="\r")
        
        # 批量shuffle TCR
        tcr_idx = rng.permutation(n)
        neg = pos.copy()
        neg[COL_TRA] = pos[COL_TRA].iloc[tcr_idx].reset_index(drop=True).values
        neg[COL_TRB] = pos[COL_TRB].iloc[tcr_idx].reset_index(drop=True).values
        
        # 批量检查并修正重复配对
        needs_resample = np.zeros(n, dtype=bool)
        
        for i in range(n):
            tra = str(neg.at[i, COL_TRA]) if pd.notna(neg.at[i, COL_TRA]) else "NA"
            trb = str(neg.at[i, COL_TRB]) if pd.notna(neg.at[i, COL_TRB]) else "NA"
            epi = str(neg.at[i, COL_EPI]) if pd.notna(neg.at[i, COL_EPI]) else "NA"
            mhca = str(neg.at[i, COL_MHCA]) if pd.notna(neg.at[i, COL_MHCA]) else "NA"
            mhcb = str(neg.at[i, COL_MHCB]) if pd.notna(neg.at[i, COL_MHCB]) else "NA"
            
            pair_id = f"{tra}|{trb}|{epi}|{mhca}|{mhcb}"
            
            if pair_id in original_pairs:
                needs_resample[i] = True
        
        # 批量重采样
        if needs_resample.any():
            resample_indices = np.where(needs_resample)[0]
            for i in resample_indices:
                tries = 0
                while tries < MAX_RESAMPLE_TRIES:
                    new_tcr_idx = rng.choice(n)
                    new_tra = pos.at[new_tcr_idx, COL_TRA]
                    new_trb = pos.at[new_tcr_idx, COL_TRB]
                    
                    # 检查新组合是否不同
                    tra_str = str(new_tra) if pd.notna(new_tra) else "NA"
                    trb_str = str(new_trb) if pd.notna(new_trb) else "NA"
                    epi = str(neg.at[i, COL_EPI]) if pd.notna(neg.at[i, COL_EPI]) else "NA"
                    mhca = str(neg.at[i, COL_MHCA]) if pd.notna(neg.at[i, COL_MHCA]) else "NA"
                    mhcb = str(neg.at[i, COL_MHCB]) if pd.notna(neg.at[i, COL_MHCB]) else "NA"
                    
                    new_pair_id = f"{tra_str}|{trb_str}|{epi}|{mhca}|{mhcb}"
                    
                    if new_pair_id not in original_pairs:
                        neg.at[i, COL_TRA] = new_tra
                        neg.at[i, COL_TRB] = new_trb
                        break
                    tries += 1
        
        neg[COL_Y] = 0
        neg_blocks.append(neg)
    
    print()  # 换行
    if neg_blocks:
        neg_df = pd.concat(neg_blocks, axis=0, ignore_index=True)
    else:
        neg_df = pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})
    
    return neg_df

def gen_neg_healthy_sample_fast(pos: pd.DataFrame, healthy_pool: pd.DataFrame, mult: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    快速版本：healthy_sample阴性生成
    """
    n = len(pos)
    if n == 0 or mult <= 0:
        return pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})

    original_pairs = get_original_pairs_fast(pos)
    
    # 处理健康TCR库
    healthy_pool = healthy_pool.copy()
    healthy_pool[COL_TRA] = healthy_pool[COL_TRA].where(pd.notna(healthy_pool[COL_TRA]), None)
    healthy_pool[COL_TRB] = healthy_pool[COL_TRB].where(pd.notna(healthy_pool[COL_TRB]), None)
    
    # 移除两条链都为NA的行
    both_na = healthy_pool[COL_TRA].isna() & healthy_pool[COL_TRB].isna()
    healthy_pool = healthy_pool[~both_na].reset_index(drop=True)
    
    if len(healthy_pool) == 0:
        print("  WARNING: 健康TCR库为空")
        return pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})
    
    pos = pos.reset_index(drop=True)
    total_needed = n * mult
    
    print(f"      需要采样 {total_needed:,} 个健康TCR")
    
    # 预采样健康TCR（批量采样，避免逐行检查）
    healthy_indices = rng.integers(0, len(healthy_pool), size=total_needed * 2)  # 多采样一些
    
    # 构建阴性样本
    neg_dfs = []
    
    for batch in range(mult):
        print(f"      Batch {batch+1}/{mult}", end="\r")
        
        # 复制当前批次的pMHC
        batch_start = batch * n
        batch_end = (batch + 1) * n
        pmhc_batch = pos.copy()
        
        # 为当前批次分配健康TCR
        tcr_start = batch * n
        tcr_indices = healthy_indices[tcr_start:tcr_start + n]
        
        # 应用健康TCR
        for i, tcr_idx in enumerate(tcr_indices):
            if tcr_idx < len(healthy_pool):
                sample = healthy_pool.iloc[tcr_idx]
                pmhc_batch.at[i, COL_TRA] = sample[COL_TRA]
                pmhc_batch.at[i, COL_TRB] = sample[COL_TRB]
        
        pmhc_batch[COL_Y] = 0
        neg_dfs.append(pmhc_batch)
    
    print()  # 换行
    
    if neg_dfs:
        neg_df = pd.concat(neg_dfs, axis=0, ignore_index=True)
        
        # 快速检查并移除明显的重复（可选，如果速度还是慢可以注释掉）
        print("      快速去重检查...")
        duplicate_mask = np.zeros(len(neg_df), dtype=bool)
        
        for i in range(len(neg_df)):
            if i % 10000 == 0:
                print(f"        检查进度: {i}/{len(neg_df)}", end="\r")
            
            tra = str(neg_df.at[i, COL_TRA]) if pd.notna(neg_df.at[i, COL_TRA]) else "NA"
            trb = str(neg_df.at[i, COL_TRB]) if pd.notna(neg_df.at[i, COL_TRB]) else "NA"
            epi = str(neg_df.at[i, COL_EPI]) if pd.notna(neg_df.at[i, COL_EPI]) else "NA"
            mhca = str(neg_df.at[i, COL_MHCA]) if pd.notna(neg_df.at[i, COL_MHCA]) else "NA"
            mhcb = str(neg_df.at[i, COL_MHCB]) if pd.notna(neg_df.at[i, COL_MHCB]) else "NA"
            
            pair_id = f"{tra}|{trb}|{epi}|{mhca}|{mhcb}"
            
            if pair_id in original_pairs:
                duplicate_mask[i] = True
        
        print()  # 换行
        
        if duplicate_mask.any():
            print(f"      移除 {duplicate_mask.sum()} 个重复配对")
            neg_df = neg_df[~duplicate_mask].reset_index(drop=True)
    else:
        neg_df = pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})
    
    return neg_df

# =========================
# 其他辅助函数
# =========================
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _check_required_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要列: {missing}")

def load_healthy_tcr_pool(path: str) -> pd.DataFrame:
    """加载健康TCR库"""
    ht = pd.read_csv(path)
    _check_required_cols(ht, [COL_TRA, COL_TRB], "healthy_tcr.csv")
    ht = ht[[COL_TRA, COL_TRB]].copy()
    ht[COL_TRA] = ht[COL_TRA].where(pd.notna(ht[COL_TRA]), None)
    ht[COL_TRB] = ht[COL_TRB].where(pd.notna(ht[COL_TRB]), None)
    return ht

def add_positive_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COL_Y] = 1
    return out

def check_class_balance(df, variant_name, file_name):
    pos_count = (df[COL_Y] == 1).sum()
    neg_count = (df[COL_Y] == 0).sum()
    if pos_count > 0:
        ratio = neg_count / pos_count
        print(f"  {variant_name}/{file_name}: positive={pos_count:,}, negative={neg_count:,}, ratio={ratio:.2f}x")
    else:
        print(f"  {variant_name}/{file_name}: positive=0, negative={neg_count:,}, ratio=N/A")

# =========================
# 主处理函数
# =========================
def build_and_save_variants(df_pos_raw: pd.DataFrame, healthy_pool: pd.DataFrame, in_name: str):
    rng = np.random.default_rng(RANDOM_SEED)
    _check_required_cols(df_pos_raw, [COL_TRA, COL_TRB, COL_EPI, COL_MHCA, COL_MHCB], in_name)
    
    pos = add_positive_label(df_pos_raw)
    print(f"  Processing {in_name}: {len(pos):,} positive samples")

    # 生成pmhc_not_bind阴性
    print(f"    Generating {PMHC_NEG_MULT}x pmhc_not_bind negatives (fast)...")
    neg_pmhc = gen_neg_pmhc_break_pair_fast(pos, PMHC_NEG_MULT, rng)
    print(f"      Generated {len(neg_pmhc):,} pmhc_not_bind negatives")

    # 实验一：5x pmhc_not_bind + 5x tcr_shuffle
    print(f"    Generating {EXP1_TCR_MULT}x tcr_shuffle negatives (fast)...")
    neg_tcr_shuffle = gen_neg_tcr_shuffle_fast(pos, EXP1_TCR_MULT, rng)
    exp1_shuffle = pd.concat([pos, neg_pmhc, neg_tcr_shuffle], axis=0, ignore_index=True)
    check_class_balance(exp1_shuffle, "exp1_shuffle", in_name)

    # 实验一：5x pmhc_not_bind + 5x healthy_sample  
    print(f"    Generating {EXP1_TCR_MULT}x healthy_sample negatives (fast)...")
    neg_healthy_5 = gen_neg_healthy_sample_fast(pos, healthy_pool, EXP1_TCR_MULT, rng)
    exp1_healthy = pd.concat([pos, neg_pmhc, neg_healthy_5], axis=0, ignore_index=True)
    check_class_balance(exp1_healthy, "exp1_healthy", in_name)

    # 实验二：不同倍数的healthy_sample
    exp2_variants = {}
    for m in EXP2_HEALTHY_MULTS:
        print(f"    Generating {m}x healthy_sample negatives (fast)...")
        neg_healthy_m = gen_neg_healthy_sample_fast(pos, healthy_pool, m, rng)
        exp2_variants[f"exp2_healthy_{m}x"] = pd.concat([pos, neg_pmhc, neg_healthy_m], axis=0, ignore_index=True)
        check_class_balance(exp2_variants[f"exp2_healthy_{m}x"], f"exp2_healthy_{m}x", in_name)

    # 保存
    out_map = {
        "exp1_shuffle": exp1_shuffle,
        "exp1_healthy": exp1_healthy,
        **exp2_variants,
    }

    for variant, df_out in out_map.items():
        out_dir = os.path.join(OUT_DIR, variant)
        _ensure_dir(out_dir)
        out_path = os.path.join(out_dir, in_name)
        df_out.to_csv(out_path, index=False, encoding="utf-8")

    print(f"  ✓ Completed {in_name}")

# =========================
# main
# =========================
def main():
    _ensure_dir(OUT_DIR)
    print("=" * 60)
    print("开始生成阴性样本（快速版本）")
    print("=" * 60)
    
    # 加载健康TCR库
    print(f"加载健康TCR库: {HEALTHY_TCR_CSV}")
    try:
        healthy_pool = load_healthy_tcr_pool(HEALTHY_TCR_CSV)
        print(f"  ✓ 加载完成: {len(healthy_pool):,} 个健康TCR")
    except Exception as e:
        print(f"  ✗ 加载健康TCR库失败: {e}")
        return
    
    print()

    # 处理每个文件
    for f in SPLIT_FILES:
        p = os.path.join(IN_SPLIT_DIR, f)
        if not os.path.exists(p):
            print(f"[WARN] 文件不存在: {p}")
            continue
            
        name = os.path.basename(p)
        print(f"处理文件: {name}")
        
        try:
            df = pd.read_csv(p)
            print(f"  ✓ 加载完成: {len(df):,} 行数据")
            
            build_and_save_variants(df, healthy_pool, name)
                
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
        print()

    print("=" * 60)
    print("阴性样本生成完成")
    print("=" * 60)

if __name__ == "__main__":
    main()