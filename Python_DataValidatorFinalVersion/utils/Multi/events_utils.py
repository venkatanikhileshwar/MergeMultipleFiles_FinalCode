# ================================================
# filepath: utils/Multi/events_utils.py
# ================================================
from __future__ import annotations
import pandas as pd
from typing import List

__all__ = ["to_events_long", "dedup_events"]


def to_events_long(df: pd.DataFrame, key_cols: List[str], multi_columns: List[str], source_label: str) -> pd.DataFrame:
    if not multi_columns:
        return pd.DataFrame(columns=key_cols + ["event_type", "column", "value", "__from__"])
    keep = key_cols + [c for c in multi_columns if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=key_cols + ["event_type", "column", "value", "__from__"])
    melt_cols = [c for c in multi_columns if c in df.columns]
    base = df[key_cols + melt_cols].copy()
    long = base.melt(id_vars=key_cols, value_vars=melt_cols, var_name="column", value_name="value")
    long = long[long["value"].astype(str).str.strip().astype(bool)]
    # derive a simple event_type from column name
    def _etype(c: str) -> str:
        c = str(c).lower()
        if "diagn" in c:
            return "diagnosis"
        if "med" in c or "drug" in c:
            return "medication"
        if "allerg" in c:
            return "allergy"
        if "lab" in c or "loinc" in c:
            return "lab"
        if "encounter" in c or "visit" in c:
            return "encounter"
        if "proc" in c:
            return "procedure"
        return "other"
    long["event_type"] = long["column"].map(_etype)
    long["__from__"] = source_label
    return long[key_cols + ["event_type", "column", "value", "__from__"]]


def dedup_events(events_df: pd.DataFrame, identity_cols=("event_type", "column", "value")) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    return events_df.drop_duplicates(subset=list(identity_cols + tuple(c for c in events_df.columns if c in identity_cols)))
