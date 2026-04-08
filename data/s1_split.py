# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd

# =========================
# 可调参数
# =========================
IN_CSV = "raw/4_combined_data_with_labels.csv"
OUT_DIR = "splits_meta"  # 修改输出目录以区分
RANDOM_SEED = 2025

# 分组阈值 (Strict definitions)
# Seen: count > 10
# Few:  5 <= count <= 10
# Unseen: count < 5
SEEN_MIN_THRESHOLD = 10 
FEW_MAX_THRESHOLD = 10
FEW_MIN_THRESHOLD = 5

# few-shot支持集大小
K_SHOT = 5

# seen tasks 内部 train/test 比例
SEEN_TRAIN_FRAC = 0.85 

# 保证每个task在test中至少有1个样本（当总数>=2）
MIN_TEST_PER_TASK = 1

# =========================
# 工具函数
# =========================
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

def split_seen_task_train_test(group_df: pd.DataFrame, train_frac: float, rng: np.random.Generator,
                               min_test_per_task: int = 1):
    """
    seen task 内部切 train/test
    """
    idx = group_df.index.values.copy()
    rng.shuffle(idx)
    n = len(idx)
    
    # 计算train数量
    n_tr = int(round(n * train_frac))
    n_te = n - n_tr
    
    # 保证test至少min_test_per_task个
    if n >= 2 and n_te < min_test_per_task:
        n_te = min_test_per_task
        n_tr = n - n_te
    
    # 保证train至少1个
    if n_tr < 1 and n > 0:
        n_tr = 1
        n_te = n - n_tr
    
    tr = idx[:n_tr]
    te = idx[n_tr:]
    return group_df.loc[tr], group_df.loc[te]

def random_support_query_split(group_df: pd.DataFrame, k: int, rng: np.random.Generator):
    """每个few task随机抽k条support，其余为query"""
    idx = group_df.index.values
    if len(idx) <= k:
        # 如果样本数不足k+1，全部作为query（或者根据需求调整，这里设为query）
        return group_df.iloc[0:0].copy(), group_df.copy()
    
    support_idx = rng.choice(idx, size=k, replace=False)
    support = group_df.loc[support_idx]
    query = group_df.drop(index=support_idx)
    return support, query

def get_task_sets(task_size_series, seen_thresh=10, few_max=10, few_min=5):
    """
    严格按照阈值返回三个互斥的 Task ID 集合 (Set)
    """
    tasks = task_size_series.index.tolist()
    sizes = task_size_series.values
    
    seen_set = set()
    few_set = set()
    unseen_set = set()
    
    for task, size in zip(tasks, sizes):
        if size > seen_thresh:       # > 10
            seen_set.add(task)
        elif few_min <= size <= few_max: # 5 <= size <= 10
            few_set.add(task)
        elif size < few_min:         # < 5
            unseen_set.add(task)
            
    return seen_set, few_set, unseen_set

# =========================
# 主流程
# =========================
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(">>> 开始处理数据...")
    # 1. 读取数据
    df = pd.read_csv(IN_CSV)
    
    # 2. 预处理与Task ID生成
    df["MHCA_norm"] = df["MHCA"].apply(normalize_mhc)
    df["MHCB_norm"] = df["MHCB"].apply(normalize_mhc)
    
    df["task_id"] = (
        df["epitope"].map(_fill_na_for_task) + "|" +
        df["MHCA_norm"].map(_fill_na_for_task) + "|" +
        df["MHCB_norm"].map(_fill_na_for_task)
    )
    
    # 3. 计算Task大小并划分集合
    task_sizes = df["task_id"].value_counts()
    print(f"总Task数: {len(task_sizes)}")
    
    # 获取互斥集合
    seen_ids, few_ids, unseen_ids = get_task_sets(
        task_sizes, 
        seen_thresh=SEEN_MIN_THRESHOLD, 
        few_max=FEW_MAX_THRESHOLD, 
        few_min=FEW_MIN_THRESHOLD
    )
    
    # --- 严格互斥性检查 ---
    print("\n[检查 Task 集合互斥性]")
    intersect_sf = seen_ids.intersection(few_ids)
    intersect_fu = few_ids.intersection(unseen_ids)
    intersect_su = seen_ids.intersection(unseen_ids)
    
    print(f"Seen tasks (>10): {len(seen_ids)}")
    print(f"Few tasks (5-10): {len(few_ids)}")
    print(f"Unseen tasks (<5): {len(unseen_ids)}")
    
    if intersect_sf or intersect_fu or intersect_su:
        raise ValueError(f"Task集合存在重叠! \nSF:{len(intersect_sf)}, FU:{len(intersect_fu)}, SU:{len(intersect_su)}")
    else:
        print(">>> 验证通过：三个Task集合严格互斥，无重叠。")
        
    # 4. 根据 ID 集合切分整个 DataFrame
    # 这样可以确保数据物理隔离
    df_seen_all = df[df["task_id"].isin(seen_ids)].copy()
    df_few_all = df[df["task_id"].isin(few_ids)].copy()
    df_unseen_all = df[df["task_id"].isin(unseen_ids)].copy()
    
    print(f"\n[数据行分配]")
    print(f"Total rows: {len(df)}")
    print(f"Rows in Seen group: {len(df_seen_all)}")
    print(f"Rows in Few group: {len(df_few_all)}")
    print(f"Rows in Unseen group: {len(df_unseen_all)}")
    assert len(df) == len(df_seen_all) + len(df_few_all) + len(df_unseen_all), "数据行总数不匹配，有丢失或重复！"

    # =========================
    # 5. 处理 Seen Tasks (Meta-Train / Meta-Test)
    # =========================
    meta_train_list = []
    meta_test_list = []
    
    # 这里的逻辑：
    # 如果 source 是 'EGRO_vDJDB_test'，强制放入 meta_test
    # 否则，按照比例切分 train/test
    
    for task_id, g in df_seen_all.groupby("task_id"):
        # 分离出强制测试集
        egro_mask = g["source"] == "EGRO_vDJDB_test"
        g_egro = g[egro_mask]
        g_other = g[~egro_mask]
        
        # 处理非EGRO数据：按比例切分
        if len(g_other) > 0:
            tr_df, te_df = split_seen_task_train_test(
                g_other, SEEN_TRAIN_FRAC, rng, min_test_per_task=MIN_TEST_PER_TASK
            )
            meta_train_list.append(tr_df)
            meta_test_list.append(te_df)
            
        # 处理EGRO数据：全部进测试
        if len(g_egro) > 0:
            meta_test_list.append(g_egro)

    meta_train_df = pd.concat(meta_train_list) if meta_train_list else pd.DataFrame()
    meta_test_df = pd.concat(meta_test_list) if meta_test_list else pd.DataFrame()

    # =========================
    # 6. 处理 Few Tasks (Support / Query)
    # =========================
    few_support_list = []
    few_query_list = []
    
    # 这里的逻辑：
    # 如果 source 是 'EGRO_vDJDB_test'，它必须是 query (不能作为support泄露)
    # 只有非EGRO数据可以用来选 Support
    
    for task_id, g in df_few_all.groupby("task_id"):
        egro_mask = g["source"] == "EGRO_vDJDB_test"
        g_egro = g[egro_mask]
        g_other = g[~egro_mask]
        
        # 在非EGRO数据中切分 Support/Query
        # 注意：如果非EGRO数据本身就不够K个，random_support_query_split 会全部划为Query
        supp, qry_internal = random_support_query_split(g_other, K_SHOT, rng)
        
        few_support_list.append(supp)
        few_query_list.append(qry_internal) # 内部Query
        few_query_list.append(g_egro)       # 外部强制测试集归为Query

    few_support_df = pd.concat(few_support_list) if few_support_list else pd.DataFrame()
    few_query_df = pd.concat(few_query_list) if few_query_list else pd.DataFrame()

    # =========================
    # 7. 处理 Unseen Tasks (全部 Test)
    # =========================
    unseen_test_df = df_unseen_all.copy()
    # Unseen 不需要特殊处理逻辑，因为无论是 EGRO 还是其他 source，都是 unseen test

    # =========================
    # 8. 输出与统计
    # =========================
    out_paths = {
        "meta_train.csv": meta_train_df,
        "meta_test.csv": meta_test_df,
        "few_test_support.csv": few_support_df,
        "few_test_query.csv": few_query_df,
        "unseen_test.csv": unseen_test_df,
    }
    
    print("\n" + "="*50)
    print("最终数据集保存:")
    for name, d in out_paths.items():
        p = os.path.join(OUT_DIR, name)
        # 确保列顺序一致(可选)
        d.to_csv(p, index=False, encoding="utf-8")
        print(f"[OK] {name:<20} : {len(d):>6,} rows | Tasks: {d['task_id'].nunique()}")

    # 最终完整性检查
    total_out = sum(len(d) for d in out_paths.values())
    print("-" * 50)
    print(f"原始数据行数: {len(df)}")
    print(f"输出数据行数: {total_out}")
    if len(df) == total_out:
        print(">>> 完美匹配：所有数据已正确分配，无遗漏无重复。")
    else:
        print(f"!!! 警告：存在数据差异 {len(df) - total_out} 行")

if __name__ == "__main__":
    main()