# -*- coding: utf-8 -*-
"""
Generate negative-background variants of each split file.

Negative construction (per positive record):
  - 5x "pmhc_not_bind": the TCR is kept and the pMHC (epitope + MHC) is replaced
    by a randomly permuted one; because the pMHC changes, task_id is rebuilt as
    epitope|MHCA_norm|MHCB_norm and the raw MHC columns are re-sampled together
    with the normalized ones so that the two stay consistent. Any generated
    record whose (cdr3_TRA, cdr3_TRB, epitope, MHCA_norm, MHCB_norm) tuple
    matches a known positive is re-sampled (up to MAX_RESAMPLE_TRIES times).
  - Nx "healthy_sample" (N in {10, 100, 1000}): the pMHC is kept and the TCR is
    replaced by a TCR drawn (with replacement) from a healthy-donor background
    pool. The pMHC does not change, so task_id stays valid. Known-positive
    tuple collisions are removed.

Output: splits_neg/{10x,100x,1000x}/<file>.csv
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = int(os.environ.get("S2_SEED", "2025"))
IN_SPLIT_DIR = os.environ.get("S2_IN_DIR", "splits")
OUT_DIR = os.environ.get("S2_OUT_DIR", "splits_neg")
HEALTHY_TCR_CSV = os.environ.get("S2_HEALTHY_TCR", "raw/healthy_hum/healthy_tcr.csv")
_ALL_SPLIT_FILES = [
    "meta_train.csv", "meta_val.csv", "meta_test.csv",
    "few_test_support.csv", "few_test_query.csv", "unseen_test.csv"
]
_env_files = os.environ.get("S2_FILES", "").strip()
SPLIT_FILES = [f for f in _ALL_SPLIT_FILES if not _env_files or f in _env_files.split(",")]

COL_TRA = "cdr3_TRA"
COL_TRB = "cdr3_TRB"
COL_EPI = "epitope"
COL_MHCA = "MHCA_norm"
COL_MHCB = "MHCB_norm"
COL_Y = "Target"
COL_TASK = "task_id"
COL_MHCA_RAW = "MHCA"
COL_MHCB_RAW = "MHCB"

def _fill_na_for_task(v):
    return str(v) if pd.notna(v) else "NA"

def rebuild_task_id(df: pd.DataFrame) -> pd.Series:
    """task_id = epitope|MHCA_norm|MHCB_norm (NA filled), same rule as split_tasks.py."""
    return (df[COL_EPI].map(_fill_na_for_task) + "|" +
            df[COL_MHCA].map(_fill_na_for_task) + "|" +
            df[COL_MHCB].map(_fill_na_for_task))

def build_pair_ids(df: pd.DataFrame) -> pd.Series:
    """Vectorized five-field pair identifier."""
    def col(c):
        return df[c].where(df[c].notna(), "NA").astype(str)
    return (col(COL_TRA) + "|" + col(COL_TRB) + "|" + col(COL_EPI) + "|" +
            col(COL_MHCA) + "|" + col(COL_MHCB))

PMHC_NEG_MULT = 5
EXP2_HEALTHY_MULTS = [10, 100, 1000]
MAX_RESAMPLE_TRIES = 10


def get_original_pairs_fast(pos: pd.DataFrame):
    identifiers = []
    for idx, row in pos.iterrows():
        tra = str(row[COL_TRA]) if pd.notna(row[COL_TRA]) else "NA"
        trb = str(row[COL_TRB]) if pd.notna(row[COL_TRB]) else "NA"
        epi = str(row[COL_EPI]) if pd.notna(row[COL_EPI]) else "NA"
        mhca = str(row[COL_MHCA]) if pd.notna(row[COL_MHCA]) else "NA"
        mhcb = str(row[COL_MHCB]) if pd.notna(row[COL_MHCB]) else "NA"
        identifiers.append(f"{tra}|{trb}|{epi}|{mhca}|{mhcb}")
    return set(identifiers)


def gen_neg_pmhc_break_pair_fast(pos: pd.DataFrame, mult: int, rng: np.random.Generator) -> pd.DataFrame:
    """pmhc_not_bind negatives: keep the TCR, replace the pMHC."""
    n = len(pos)
    if n == 0 or mult <= 0:
        return pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})

    original_pairs = get_original_pairs_fast(pos)
    pos = pos.reset_index(drop=True)

    neg_blocks = []

    for batch in range(mult):
        print(f"      Batch {batch+1}/{mult}", end="\r")

        epi_idx = rng.permutation(n)
        mhc_idx = rng.permutation(n)

        neg = pos.copy()
        neg[COL_EPI] = pos[COL_EPI].iloc[epi_idx].reset_index(drop=True).values
        neg[COL_MHCA] = pos[COL_MHCA].iloc[mhc_idx].reset_index(drop=True).values
        neg[COL_MHCB] = pos[COL_MHCB].iloc[mhc_idx].reset_index(drop=True).values
        # keep the raw and normalized MHC columns consistent with each other
        has_raw_mhc = COL_MHCA_RAW in pos.columns and COL_MHCB_RAW in pos.columns
        if has_raw_mhc:
            neg[COL_MHCA_RAW] = pos[COL_MHCA_RAW].iloc[mhc_idx].reset_index(drop=True).values
            neg[COL_MHCB_RAW] = pos[COL_MHCB_RAW].iloc[mhc_idx].reset_index(drop=True).values

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

        if needs_resample.any():
            resample_indices = np.where(needs_resample)[0]
            for i in resample_indices:
                tries = 0
                while tries < MAX_RESAMPLE_TRIES:
                    new_epi_idx = rng.choice(n)
                    new_mhc_idx = rng.choice(n)

                    new_epi = pos.at[new_epi_idx, COL_EPI]
                    new_mhca = pos.at[new_mhc_idx, COL_MHCA]
                    new_mhcb = pos.at[new_mhc_idx, COL_MHCB]

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
                        if has_raw_mhc:
                            neg.at[i, COL_MHCA_RAW] = pos.at[new_mhc_idx, COL_MHCA_RAW]
                            neg.at[i, COL_MHCB_RAW] = pos.at[new_mhc_idx, COL_MHCB_RAW]
                        break
                    tries += 1

        neg[COL_Y] = 0
        # the pMHC changed, so task_id must be rebuilt from the new epitope|MHC
        if COL_TASK in neg.columns:
            neg[COL_TASK] = rebuild_task_id(neg)
        neg_blocks.append(neg)

    print()
    if neg_blocks:
        neg_df = pd.concat(neg_blocks, axis=0, ignore_index=True)
    else:
        neg_df = pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})

    return neg_df


def gen_neg_tcr_shuffle_fast(pos: pd.DataFrame, mult: int, rng: np.random.Generator) -> pd.DataFrame:
    """tcr_shuffle negatives: keep the pMHC, replace the TCR with another
    TCR from the positive set (permutation)."""
    n = len(pos)
    if n == 0 or mult <= 0:
        return pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})

    original_pairs = get_original_pairs_fast(pos)
    pos = pos.reset_index(drop=True)

    neg_blocks = []

    for batch in range(mult):
        print(f"      Batch {batch+1}/{mult}", end="\r")

        tcr_idx = rng.permutation(n)
        neg = pos.copy()
        neg[COL_TRA] = pos[COL_TRA].iloc[tcr_idx].reset_index(drop=True).values
        neg[COL_TRB] = pos[COL_TRB].iloc[tcr_idx].reset_index(drop=True).values

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

        if needs_resample.any():
            resample_indices = np.where(needs_resample)[0]
            for i in resample_indices:
                tries = 0
                while tries < MAX_RESAMPLE_TRIES:
                    new_tcr_idx = rng.choice(n)
                    new_tra = pos.at[new_tcr_idx, COL_TRA]
                    new_trb = pos.at[new_tcr_idx, COL_TRB]

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

    print()
    if neg_blocks:
        neg_df = pd.concat(neg_blocks, axis=0, ignore_index=True)
    else:
        neg_df = pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})

    return neg_df


def gen_neg_healthy_sample_fast(pos: pd.DataFrame, healthy_pool: pd.DataFrame, mult: int, rng: np.random.Generator) -> pd.DataFrame:
    """healthy_sample negatives: keep the pMHC, replace the TCR with a TCR
    drawn with replacement from the healthy-donor background pool."""
    n = len(pos)
    if n == 0 or mult <= 0:
        return pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})

    original_pairs = get_original_pairs_fast(pos)

    healthy_pool = healthy_pool.copy()
    healthy_pool[COL_TRA] = healthy_pool[COL_TRA].where(pd.notna(healthy_pool[COL_TRA]), None)
    healthy_pool[COL_TRB] = healthy_pool[COL_TRB].where(pd.notna(healthy_pool[COL_TRB]), None)

    both_na = healthy_pool[COL_TRA].isna() & healthy_pool[COL_TRB].isna()
    healthy_pool = healthy_pool[~both_na].reset_index(drop=True)

    if len(healthy_pool) == 0:
        print("  WARNING: healthy TCR pool is empty")
        return pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})

    pos = pos.reset_index(drop=True)
    total_needed = n * mult

    print(f"      Sampling {total_needed:,} healthy TCRs")

    healthy_indices = rng.integers(0, len(healthy_pool), size=total_needed * 2)

    neg_dfs = []

    for batch in range(mult):
        print(f"      Batch {batch+1}/{mult}", end="\r")

        pmhc_batch = pos.copy()
        tcr_start = batch * n
        tcr_indices = healthy_indices[tcr_start:tcr_start + n]
        sampled = healthy_pool.iloc[tcr_indices]
        pmhc_batch[COL_TRA] = sampled[COL_TRA].to_numpy()
        pmhc_batch[COL_TRB] = sampled[COL_TRB].to_numpy()

        pmhc_batch[COL_Y] = 0
        neg_dfs.append(pmhc_batch)

    print()

    if neg_dfs:
        neg_df = pd.concat(neg_dfs, axis=0, ignore_index=True)
        pair_ids = build_pair_ids(neg_df)
        duplicate_mask = pair_ids.isin(original_pairs).to_numpy()
        if duplicate_mask.any():
            print(f"      Removing {duplicate_mask.sum()} known-positive collisions")
            neg_df = neg_df[~duplicate_mask].reset_index(drop=True)
    else:
        neg_df = pd.DataFrame(columns=pos.columns).assign(**{COL_Y: 0})

    return neg_df


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _check_required_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def load_healthy_tcr_pool(path: str) -> pd.DataFrame:
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
        print(f"  {variant_name}/{file_name}: positive={pos_count:,}, negative={neg_count:,}, "
              f"ratio={neg_count / pos_count:.2f}x")


def build_and_save_variants(df_pos_raw: pd.DataFrame, healthy_pool: pd.DataFrame, in_name: str):
    rng = np.random.default_rng(RANDOM_SEED)
    _check_required_cols(df_pos_raw, [COL_TRA, COL_TRB, COL_EPI, COL_MHCA, COL_MHCB], in_name)

    pos = add_positive_label(df_pos_raw)
    print(f"  Processing {in_name}: {len(pos):,} positive samples")

    print(f"    Generating {PMHC_NEG_MULT}x pmhc_not_bind negatives...")
    neg_pmhc = gen_neg_pmhc_break_pair_fast(pos, PMHC_NEG_MULT, rng)
    print(f"      Generated {len(neg_pmhc):,} pmhc_not_bind negatives")

    out_map = {}
    for m in EXP2_HEALTHY_MULTS:
        print(f"    Generating {m}x healthy_sample negatives...")
        neg_healthy_m = gen_neg_healthy_sample_fast(pos, healthy_pool, m, rng)
        df_m = pd.concat([pos, neg_pmhc, neg_healthy_m], axis=0, ignore_index=True)
        check_class_balance(df_m, f"{m}x", in_name)
        out_map[f"{m}x"] = df_m

    for variant, df_out in out_map.items():
        out_dir = os.path.join(OUT_DIR, variant)
        _ensure_dir(out_dir)
        df_out.to_csv(os.path.join(out_dir, in_name), index=False, encoding="utf-8")

    print(f"  Done: {in_name}")


def main():
    _ensure_dir(OUT_DIR)
    print("=" * 60)
    print("Generating negative backgrounds")
    print("=" * 60)

    print(f"Loading healthy TCR pool: {HEALTHY_TCR_CSV}")
    try:
        healthy_pool = load_healthy_tcr_pool(HEALTHY_TCR_CSV)
        print(f"  Loaded: {len(healthy_pool):,} healthy TCRs")
    except Exception as e:
        print(f"  Failed to load healthy TCR pool: {e}")
        return

    print()

    for f in SPLIT_FILES:
        p = os.path.join(IN_SPLIT_DIR, f)
        if not os.path.exists(p):
            print(f"[WARN] file not found: {p}")
            continue

        name = os.path.basename(p)
        print(f"Processing file: {name}")

        try:
            df = pd.read_csv(p)
            print(f"  Loaded: {len(df):,} rows")
            build_and_save_variants(df, healthy_pool, name)
        except Exception as e:
            print(f"  Failed: {e}")
        print()

    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
