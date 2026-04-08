# -*- coding: utf-8 -*-
import os
import re
import pandas as pd

# ====== 可改参数 ======
IN_DIR = "."   # 三个csv所在目录；如果就在当前目录，改成 "."
FILES = [
    "unseen_test.csv",
    "meta_test.csv",
]
OUT_CSV = "merged_tests.csv"

# 需要保留的列
KEEP_COLS = ["cdr3_TRA", "cdr3_TRB", "epitope", "MHCA", "MHCB", "Target"]


def load_and_select(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] {os.path.basename(csv_path)} 缺少列: {missing}\n"
                         f"当前列名: {list(df.columns)}")

    df = df[KEEP_COLS].copy()
    df = df.rename(columns={"Target": "label"})
    return df

def main():
    dfs = []
    for fn in FILES:
        path = os.path.join(IN_DIR, fn)
        cur = load_and_select(path)
        # 如果你想保留来源split信息，可以取消下一行注释
        # cur["split"] = os.path.splitext(fn)[0]
        dfs.append(cur)

    merged = pd.concat(dfs, axis=0, ignore_index=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] merged rows = {len(merged):,}")
    print(f"[OK] saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()
