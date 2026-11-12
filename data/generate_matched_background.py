# -*- coding: utf-8 -*-
"""
Generate chain-status- and length-matched negative backgrounds.

For each split file of positives, one matched variant is produced:
  positives + 5x pmhc_not_bind + Nx chain-matched healthy negatives

Matching rules for the healthy negatives:
  - the healthy pool is stratified into three chain-status layers
    (paired / TRA_only / TRB_only);
  - each positive is assigned negatives sampled (with replacement) from its own
    chain-status layer;
  - within a layer, CDR3 total length is matched by nearest length:
    |Δ| <= 4, then |Δ| <= 8, then any record in the layer;
  - if a layer is empty, sampling falls back to the whole pool (counted in the
    report);
  - if a generated record's (cdr3_TRA, cdr3_TRB, epitope, MHCA_norm, MHCB_norm)
    tuple matches a known positive, it is re-sampled (up to 10 tries);
  - the pMHC is unchanged, so task_id stays valid.

Output: splits_matched/<tag or matched{N}x>/<file>.csv
Report: <out_dir>/match_report.md
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import generate_negatives as GEN

IN_SPLIT_DIR = os.environ.get("M_IN_DIR", os.path.join(HERE, "splits"))
HEALTHY_TCR_CSV = os.environ.get(
    "M_HEALTHY_TCR",
    os.path.join(HERE, "raw", "healthy_hum", "healthy_tcr_extended.csv"))
OUT_BASE = os.environ.get("M_OUT_DIR", os.path.join(HERE, "splits_matched"))

SEED = int(os.environ.get("M_SEED", "2025"))
PMHC_MULT = int(os.environ.get("M_PMHC_MULT", "5"))
MATCH_MULT = int(os.environ.get("M_MATCH_MULT", "10"))
TAG = os.environ.get("M_TAG", "")
MAX_RESAMPLE_TRIES = 10
TIGHT_DIFF = 4
RELAXED_DIFF = 8

FILES_ALL = ["meta_train.csv", "meta_test.csv", "unseen_test.csv",
             "few_test_support.csv", "few_test_query.csv"]
FILES = os.environ.get("M_FILES", ",".join(FILES_ALL)).split(",")

ARM_NAME = TAG if TAG else f"matched{MATCH_MULT}x"
REPORT_PATH = os.path.join(OUT_BASE, "match_report.md")

COL_TRA, COL_TRB, COL_EPI = GEN.COL_TRA, GEN.COL_TRB, GEN.COL_EPI
COL_MHCA, COL_MHCB, COL_Y, COL_TASK = GEN.COL_MHCA, GEN.COL_MHCB, GEN.COL_Y, GEN.COL_TASK


def chain_status_arrays(df):
    """Return (status, total CDR3 length); status in {paired, TRA_only, TRB_only, both_na}."""
    tra = df[COL_TRA]
    trb = df[COL_TRB]
    has_a = tra.notna() & (tra.astype(str).str.len() > 0)
    has_b = trb.notna() & (trb.astype(str).str.len() > 0)
    status = np.full(len(df), "both_na", dtype=object)
    status[has_a.to_numpy() & has_b.to_numpy()] = "paired"
    status[has_a.to_numpy() & ~has_b.to_numpy()] = "TRA_only"
    status[~has_a.to_numpy() & has_b.to_numpy()] = "TRB_only"
    la = tra.where(has_a, "").astype(str).str.len().to_numpy()
    lb = trb.where(has_b, "").astype(str).str.len().to_numpy()
    return status, la + lb


def gen_neg_healthy_matched(pos, pool_status, pool_len, mult, rng, pos_pair_ids):
    """Sample chain-matched healthy negatives. Returns (neg_df, stats)."""
    n = len(pos)
    pos = pos.reset_index(drop=True)
    pos_status, pos_len = chain_status_arrays(pos)

    pool_size = len(pool_status)
    pool_idx_all = np.arange(pool_size)
    layer_idx = {s: pool_idx_all[pool_status == s] for s in ("paired", "TRA_only", "TRB_only")}

    stats = {
        "stage_tight": 0, "stage_relaxed": 0, "stage_layer_any": 0,
        "empty_layer_fallback": {"paired": 0, "TRA_only": 0, "TRB_only": 0},
        "len_diffs": [],
        "resample_collisions": 0,
    }

    cand_cache = {}

    def candidates_for(layer, target_len):
        key = (layer, target_len)
        if key in cand_cache:
            return cand_cache[key]
        idx = layer_idx[layer]
        if len(idx) == 0:
            result = ("empty", pool_idx_all)
        else:
            d = np.abs(pool_len[idx].astype(np.int64) - int(target_len))
            c4 = idx[d <= TIGHT_DIFF]
            if len(c4) > 0:
                result = ("tight", c4)
            else:
                c8 = idx[d <= RELAXED_DIFF]
                if len(c8) > 0:
                    result = ("relaxed", c8)
                else:
                    result = ("layer_any", idx)
        cand_cache[key] = result
        return result

    neg_blocks = []
    req_meta = []

    for batch in range(mult):
        print(f"      Matched batch {batch+1}/{mult}", end="\r")
        neg = pos.copy()
        sampled_rows = np.empty(n, dtype=np.int64)
        for i in range(n):
            layer = pos_status[i]
            target = pos_len[i]
            stage, cands = candidates_for(layer, target)
            if stage == "empty":
                stats["empty_layer_fallback"][layer] += 1
            elif stage == "tight":
                stats["stage_tight"] += 1
            elif stage == "relaxed":
                stats["stage_relaxed"] += 1
            else:
                stats["stage_layer_any"] += 1
            sampled_rows[i] = cands[rng.integers(0, len(cands))]
            req_meta.append((target, layer))
        neg[COL_TRA] = pool_tra[sampled_rows]
        neg[COL_TRB] = pool_trb[sampled_rows]
        neg_blocks.append(neg)

    print()
    neg_df = pd.concat(neg_blocks, axis=0, ignore_index=True)

    # re-sample any record colliding with a known positive pair
    pair_ids = GEN.build_pair_ids(neg_df)
    collision = pair_ids.isin(pos_pair_ids).to_numpy()
    if collision.any():
        stats["resample_collisions"] = int(collision.sum())
        for i in np.where(collision)[0]:
            target, layer = req_meta[i]
            stage, cands = candidates_for(layer, target)
            for _ in range(MAX_RESAMPLE_TRIES):
                j = cands[rng.integers(0, len(cands))]
                neg_df.at[i, COL_TRA] = pool_tra[j]
                neg_df.at[i, COL_TRB] = pool_trb[j]
                pid = GEN.build_pair_ids(neg_df.iloc[[i]]).iloc[0]
                if pid not in pos_pair_ids:
                    break

    final_len = chain_status_arrays(neg_df)[1]
    targets = np.array([m[0] for m in req_meta], dtype=np.int64)
    stats["len_diffs"] = np.abs(final_len.astype(np.int64) - targets)

    neg_df[COL_Y] = 0
    return neg_df, stats


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(os.path.join(OUT_BASE, ARM_NAME), exist_ok=True)

    global pool_tra, pool_trb
    print(f"Loading healthy TCR pool: {HEALTHY_TCR_CSV}")
    pool = GEN.load_healthy_tcr_pool(HEALTHY_TCR_CSV)
    both_na = pool[COL_TRA].isna() & pool[COL_TRB].isna()
    pool = pool[~both_na].reset_index(drop=True)
    pool_status, pool_len = chain_status_arrays(pool)
    pool_tra = pool[COL_TRA].to_numpy(dtype=object)
    pool_trb = pool[COL_TRB].to_numpy(dtype=object)
    print(f"  Pool: {len(pool):,} records; paired={int((pool_status=='paired').sum()):,}, "
          f"TRA_only={int((pool_status=='TRA_only').sum()):,}, "
          f"TRB_only={int((pool_status=='TRB_only').sum()):,}")

    report = {
        "pool": {
            "total": int(len(pool)),
            "paired": int((pool_status == "paired").sum()),
            "TRA_only": int((pool_status == "TRA_only").sum()),
            "TRB_only": int((pool_status == "TRB_only").sum()),
        },
        "files": {},
    }

    for fname in FILES:
        path = os.path.join(IN_SPLIT_DIR, fname)
        df_raw = pd.read_csv(path)
        rng = np.random.default_rng(SEED)
        pos = GEN.add_positive_label(df_raw)
        n_pos = len(pos)
        print(f"\nProcessing {fname}: {n_pos:,} positives")

        print(f"    Generating {PMHC_MULT}x pmhc_not_bind negatives...")
        neg_pmhc = GEN.gen_neg_pmhc_break_pair_fast(pos, PMHC_MULT, rng)

        print(f"    Generating {MATCH_MULT}x chain-matched healthy negatives...")
        pos_pair_ids = set(GEN.build_pair_ids(pos))
        neg_matched, mstats = gen_neg_healthy_matched(
            pos, pool_status, pool_len, MATCH_MULT, rng, pos_pair_ids)
        arm_df = pd.concat([pos, neg_pmhc, neg_matched], axis=0, ignore_index=True)
        arm_df.to_csv(os.path.join(OUT_BASE, ARM_NAME, fname), index=False)

        fstat = {"n_pos": n_pos, "rows": int(len(arm_df))}

        rebuilt = GEN.rebuild_task_id(arm_df)
        fstat["task_id_mismatch"] = int((rebuilt != arm_df[COL_TASK].astype(str)).sum())

        pos_status, _ = chain_status_arrays(pos)
        neg_status, _ = chain_status_arrays(neg_matched)

        def dist(s):
            return {k: round(float((s == k).mean()), 4) for k in ("paired", "TRA_only", "TRB_only")} \
                if len(s) else {}
        fstat["pos_chain_dist"] = dist(pos_status)
        fstat["matched_neg_chain_dist"] = dist(neg_status)

        ld = mstats["len_diffs"]
        fstat["len_diff"] = {
            "mean": round(float(ld.mean()), 3),
            "median": float(np.median(ld)),
            "p90": float(np.percentile(ld, 90)),
            "pct_le4": round(float((ld <= TIGHT_DIFF).mean()), 4),
            "pct_le8": round(float((ld <= RELAXED_DIFF).mean()), 4),
        }
        fstat["fallback"] = {
            "stage_tight": mstats["stage_tight"],
            "stage_relaxed": mstats["stage_relaxed"],
            "stage_layer_any": mstats["stage_layer_any"],
            "empty_layer_fallback": mstats["empty_layer_fallback"],
            "resample_collisions": mstats["resample_collisions"],
        }

        report["files"][fname] = fstat
        print(f"    rows: {len(arm_df):,}  task_id_mismatch={fstat['task_id_mismatch']}")

    write_report(report)
    print(f"\nReport written to: {REPORT_PATH}")


def write_report(report):
    L = []
    L.append("# Chain-matched negative background generation report\n")
    L.append(f"- Random seed: {SEED} (one `np.random.default_rng({SEED})` per input file)")
    L.append(f"- Input: `{IN_SPLIT_DIR}` ({len(FILES)} split files)")
    L.append(f"- Output: `{OUT_BASE}/{ARM_NAME}/`")
    L.append(f"- Matching rules: healthy pool stratified by chain status; sampling with "
             f"replacement from the positive's own layer; within-layer nearest length "
             f"(|d|<={TIGHT_DIFF} -> <={RELAXED_DIFF} -> any in layer); empty layer -> "
             f"whole pool (counted); known-positive tuple collisions re-sampled up to "
             f"{MAX_RESAMPLE_TRIES} times; task_id unchanged\n")

    p = report["pool"]
    L.append("## Healthy pool chain-status layers\n")
    L.append("| Layer | Records | Fraction |")
    L.append("|---|---|---|")
    for k in ("paired", "TRA_only", "TRB_only"):
        L.append(f"| {k} | {p[k]:,} | {p[k]/p['total']:.4f} |")
    L.append(f"| Total | {p['total']:,} | 1.0000 |\n")

    L.append("## Rows per file and task_id consistency\n")
    L.append("| File | Positives | Rows / expected | task_id mismatches |")
    L.append("|---|---|---|---|")
    for f in FILES:
        s = report["files"][f]
        exp = s["n_pos"] * (1 + PMHC_MULT + MATCH_MULT)
        L.append(f"| {f} | {s['n_pos']:,} | {s['rows']:,} / {exp:,} | {s['task_id_mismatch']} |")
    L.append("")

    L.append("## Chain-status distribution: positives vs matched negatives\n")
    L.append("| File | Positive paired/TRA/TRB | Matched-negative paired/TRA/TRB |")
    L.append("|---|---|---|")
    for f in FILES:
        s = report["files"][f]
        fmt = lambda d: f"{d['paired']:.4f} / {d['TRA_only']:.4f} / {d['TRB_only']:.4f}"
        L.append(f"| {f} | {fmt(s['pos_chain_dist'])} | {fmt(s['matched_neg_chain_dist'])} |")
    L.append("")

    L.append("## Length difference |sampled length - positive length|\n")
    L.append(f"| File | mean | median | p90 | fraction <={TIGHT_DIFF} | fraction <={RELAXED_DIFF} |")
    L.append("|---|---|---|---|---|---|")
    for f in FILES:
        d = report["files"][f]["len_diff"]
        L.append(f"| {f} | {d['mean']} | {d['median']} | {d['p90']} | {d['pct_le4']} | {d['pct_le8']} |")
    L.append("")

    L.append("## Matching stages and fallback counts\n")
    L.append(f"| File | tight (<={TIGHT_DIFF}) | relaxed (<={RELAXED_DIFF}) | any-in-layer | empty-layer fallback | collision re-samples |")
    L.append("|---|---|---|---|---|---|")
    for f in FILES:
        fb = report["files"][f]["fallback"]
        empty = fb["empty_layer_fallback"]
        empty_str = ", ".join(f"{k}:{v}" for k, v in empty.items()) if any(empty.values()) else "0"
        L.append(f"| {f} | {fb['stage_tight']:,} | {fb['stage_relaxed']:,} | {fb['stage_layer_any']:,} | "
                 f"{empty_str} | {fb['resample_collisions']} |")
    L.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
