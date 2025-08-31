# utils/mapping_utils.py
from __future__ import annotations
import pandas as pd

def _base_name(col: str) -> str:
    """Return the header portion after 'Sheet::Header' (case/space-insensitive key)."""
    return (col.split("::", 1)[1] if "::" in col else col).strip().lower()


def expand_mapping_across_sheets(remap: dict, all_file_cols: list) -> dict:
    """
    If the user maps 'Sheet1::Amt' → 'amount', propagate that mapping to every
    other 'X::Amt' across sheets (case/space-insensitive on the base name).
    Does not overwrite explicit user choices.
    """
    by_base = {}
    for c in all_file_cols:
        by_base.setdefault(_base_name(c), []).append(c)

    expanded = dict(remap)
    for src, target in remap.items():
        if not target or target == "— Ignore —":
            continue
        b = _base_name(src)
        for col in by_base.get(b, []):
            if col not in expanded:
                expanded[col] = target
    return expanded


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicate-named columns row-wise (first non-null wins).
    This is required when multiple sheets map to the same DB column
    and after renaming you end up with duplicate column names.

    Example:
        ['record_id','record_id','amount','amount','status'] → collapse to
        ['record_id','amount','status'] keeping the first non-null per row.
    """
    if df is None or df.empty:
        return df

    # Work on a copy to avoid inplace shape-change issues
    out = df.copy()
    # Keep order of first appearances
    unique_names_in_order = list(dict.fromkeys(out.columns))

    for name in unique_names_in_order:
        dupes = [c for c in out.columns if c == name]
        if len(dupes) <= 1:
            continue

        # Merge duplicates: first non-null across those duplicate columns
        merged = out[dupes].bfill(axis=1).iloc[:, 0]

        # Remove all duplicates of that name, then add a single column back
        out = out.drop(columns=dupes)
        out[name] = merged

        # Reorder to place the merged column where the first duplicate was
        # (optional – keeps columns visually stable)
        first_pos = unique_names_in_order.index(name)
        cols = list(out.columns)
        # Move 'name' to desired position
        cols.remove(name)
        cols.insert(first_pos, name)
        out = out.loc[:, cols]

    return out
