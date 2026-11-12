# -*- coding: utf-8 -*-
"""
Split the combined positive TCR-pMHC dataset into task-disjoint partitions.

Task definition: task_id = epitope|MHCA_norm|MHCB_norm (normalized MHC names).

Partition rules (by number of positive records per task):
  - seen   : count > 10   -> split into meta_train (85%) / meta_test (15%)
  - few    : 5 <= count <= 10 -> per task, 5 records form the support set,
             the rest form the query set
  - unseen : count < 5    -> all records go to unseen_test

Records with source == "EGRO_vDJDB_test" are always held out: they go to
meta_test (seen tasks) or few_test_query (few tasks) and are never used for
training or support.

Output: splits/{meta_train,meta_test,few_test_support,few_test_query,unseen_test}.csv
"""

import os
import re
import numpy as np
import pandas as pd

IN_CSV = os.environ.get("S1_IN_CSV", "raw/4_combined_data_with_labels.csv")
OUT_DIR = os.environ.get("S1_OUT_DIR", "splits")
RANDOM_SEED = int(os.environ.get("S1_SEED", "2025"))

SEEN_MIN_THRESHOLD = 10
FEW_MAX_THRESHOLD = 10
FEW_MIN_THRESHOLD = 5

K_SHOT = 5
SEEN_TRAIN_FRAC = 0.85
MIN_TEST_PER_TASK = 1


def normalize_mhc(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return x
    s = s.replace("*", "_").replace(":", "")
    s = re.sub(r"\s+", "", s)
    return s

def _fill_na_for_task(x):
    return "NA" if pd.isna(x) else str(x)

def split_seen_task_train_test(group_df, train_frac, rng, min_test_per_task=1):
    idx = group_df.index.values.copy()
    rng.shuffle(idx)
    n = len(idx)

    n_tr = int(round(n * train_frac))
    n_te = n - n_tr

    if n >= 2 and n_te < min_test_per_task:
        n_te = min_test_per_task
        n_tr = n - n_te

    if n_tr < 1 and n > 0:
        n_tr = 1
        n_te = n - n_tr

    return group_df.loc[idx[:n_tr]], group_df.loc[idx[n_tr:]]

def random_support_query_split(group_df, k, rng):
    """Sample k support records per few-shot task; the rest are queries."""
    idx = group_df.index.values
    if len(idx) <= k:
        return group_df.iloc[0:0].copy(), group_df.copy()

    support_idx = rng.choice(idx, size=k, replace=False)
    support = group_df.loc[support_idx]
    query = group_df.drop(index=support_idx)
    return support, query

def get_task_sets(task_size_series, seen_thresh=10, few_max=10, few_min=5):
    seen_set, few_set, unseen_set = set(), set(), set()
    for task, size in zip(task_size_series.index.tolist(), task_size_series.values):
        if size > seen_thresh:
            seen_set.add(task)
        elif few_min <= size <= few_max:
            few_set.add(task)
        else:
            unseen_set.add(task)
    return seen_set, few_set, unseen_set


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(">>> Loading data...")
    df = pd.read_csv(IN_CSV)

    df["MHCA_norm"] = df["MHCA"].apply(normalize_mhc)
    df["MHCB_norm"] = df["MHCB"].apply(normalize_mhc)

    df["task_id"] = (
        df["epitope"].map(_fill_na_for_task) + "|" +
        df["MHCA_norm"].map(_fill_na_for_task) + "|" +
        df["MHCB_norm"].map(_fill_na_for_task)
    )

    task_sizes = df["task_id"].value_counts()
    print(f"Total tasks: {len(task_sizes)}")

    seen_ids, few_ids, unseen_ids = get_task_sets(
        task_sizes,
        seen_thresh=SEEN_MIN_THRESHOLD,
        few_max=FEW_MAX_THRESHOLD,
        few_min=FEW_MIN_THRESHOLD
    )

    print("\n[Task set disjointness check]")
    print(f"Seen tasks (>10): {len(seen_ids)}")
    print(f"Few tasks (5-10): {len(few_ids)}")
    print(f"Unseen tasks (<5): {len(unseen_ids)}")
    if seen_ids & few_ids or few_ids & unseen_ids or seen_ids & unseen_ids:
        raise ValueError("Task sets overlap.")
    print(">>> OK: task sets are mutually exclusive.")

    df_seen_all = df[df["task_id"].isin(seen_ids)].copy()
    df_few_all = df[df["task_id"].isin(few_ids)].copy()
    df_unseen_all = df[df["task_id"].isin(unseen_ids)].copy()

    print(f"\n[Row assignment]")
    print(f"Total rows: {len(df)}")
    print(f"Rows in seen group: {len(df_seen_all)}")
    print(f"Rows in few group: {len(df_few_all)}")
    print(f"Rows in unseen group: {len(df_unseen_all)}")
    assert len(df) == len(df_seen_all) + len(df_few_all) + len(df_unseen_all)

    # seen tasks -> meta_train / meta_test
    meta_train_list, meta_test_list = [], []
    for task_id, g in df_seen_all.groupby("task_id"):
        egro_mask = g["source"] == "EGRO_vDJDB_test"
        g_egro = g[egro_mask]
        g_other = g[~egro_mask]

        if len(g_other) > 0:
            tr_df, te_df = split_seen_task_train_test(
                g_other, SEEN_TRAIN_FRAC, rng, min_test_per_task=MIN_TEST_PER_TASK
            )
            meta_train_list.append(tr_df)
            meta_test_list.append(te_df)

        if len(g_egro) > 0:
            meta_test_list.append(g_egro)

    meta_train_df = pd.concat(meta_train_list) if meta_train_list else pd.DataFrame()
    meta_test_df = pd.concat(meta_test_list) if meta_test_list else pd.DataFrame()

    # few tasks -> support / query
    few_support_list, few_query_list = [], []
    for task_id, g in df_few_all.groupby("task_id"):
        egro_mask = g["source"] == "EGRO_vDJDB_test"
        g_egro = g[egro_mask]
        g_other = g[~egro_mask]

        supp, qry_internal = random_support_query_split(g_other, K_SHOT, rng)
        few_support_list.append(supp)
        few_query_list.append(qry_internal)
        few_query_list.append(g_egro)

    few_support_df = pd.concat(few_support_list) if few_support_list else pd.DataFrame()
    few_query_df = pd.concat(few_query_list) if few_query_list else pd.DataFrame()

    # unseen tasks -> all to test
    unseen_test_df = df_unseen_all.copy()

    out_paths = {
        "meta_train.csv": meta_train_df,
        "meta_test.csv": meta_test_df,
        "few_test_support.csv": few_support_df,
        "few_test_query.csv": few_query_df,
        "unseen_test.csv": unseen_test_df,
    }

    print("\n" + "=" * 50)
    for name, d in out_paths.items():
        p = os.path.join(OUT_DIR, name)
        d.to_csv(p, index=False, encoding="utf-8")
        print(f"[OK] {name:<22} : {len(d):>6,} rows | tasks: {d['task_id'].nunique()}")

    total_out = sum(len(d) for d in out_paths.values())
    print("-" * 50)
    print(f"Input rows: {len(df)} | output rows: {total_out}")
    if len(df) != total_out:
        print(f"!!! WARNING: row count mismatch ({len(df) - total_out})")


if __name__ == "__main__":
    main()
