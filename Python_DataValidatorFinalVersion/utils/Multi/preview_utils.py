
# ================================================
# filepath: utils/Multi/preview_utils.py
# ================================================
from __future__ import annotations
import pandas as pd
from typing import Dict

__all__ = ["estimate_append_metrics", "estimate_join_metrics"]


def estimate_append_metrics(working_df: pd.DataFrame, incoming_df: pd.DataFrame) -> Dict:
    rows_before = len(working_df)
    rows_incoming = len(incoming_df)
    new_columns = [c for c in incoming_df.columns if c not in working_df.columns]
    # naive exact dup estimate: intersect rows (hash on tuple of values)
    try:
        dup_est = 0
        if rows_before and rows_incoming:
            lset = set(map(tuple, working_df.fillna("").values.tolist()))
            rset = set(map(tuple, incoming_df.fillna("").values.tolist()))
            dup_est = len(lset & rset)
    except Exception:
        dup_est = 0
    return {
        "rows_before": rows_before,
        "rows_incoming": rows_incoming,
        "new_columns": new_columns,
        "exact_dup_rows_estimate": dup_est,
    }


def estimate_join_metrics(working_df: pd.DataFrame, incoming_df: pd.DataFrame, key_cols: list[str]) -> Dict:
    if not key_cols:
        return {"key_overlap_pct": 0.0, "rows_after_inner_est": 0, "rows_after_left_est": len(working_df), "new_columns": [], "fanout_estimate": 0}
    left = working_df[key_cols].astype(str).fillna("").copy()
    right = incoming_df[key_cols].astype(str).fillna("").copy()
    left["__k__"] = left.apply(lambda r: "|".join(r.values.tolist()), axis=1)
    right["__k__"] = right.apply(lambda r: "|".join(r.values.tolist()), axis=1)
    lkeys = set(left["__k__"]) - {""}
    rkeys = set(right["__k__"]) - {""}
    inter = lkeys & rkeys
    key_overlap_pct = (100.0 * len(inter) / (len(lkeys) or 1)) if lkeys else 0.0
    # naive row-after estimates
    rows_after_inner_est = sum(1 for k in left["__k__"] if k in inter)
    rows_after_left_est = len(working_df)  # left keeps all anchor keys
    # fanout estimate: duplicates in right keys
    fanout_estimate = int(right["__k__"].duplicated(keep=False).sum())
    new_columns = [c for c in incoming_df.columns if c not in working_df.columns]
    return {
        "key_overlap_pct": round(key_overlap_pct, 1),
        "rows_after_inner_est": rows_after_inner_est,
        "rows_after_left_est": rows_after_left_est,
        "new_columns": new_columns,
        "fanout_estimate": fanout_estimate,
    }
