
# ================================================
# filepath: utils/Multi/provenance_utils.py
# ================================================
from __future__ import annotations
import pandas as pd
from typing import Optional

__all__ = ["add_provenance_cols", "merge_from_provenance"]


def add_provenance_cols(df: pd.DataFrame, source_file: str, source_sheet: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    if "__source__" not in out.columns:
        out["__source__"] = source_file
    else:
        out["__source__"] = out["__source__"].fillna("").astype(str).where(out["__source__"].astype(bool), source_file)
    if source_sheet is not None:
        if "__sheet__" not in out.columns:
            out["__sheet__"] = source_sheet
        else:
            out["__sheet__"] = out["__sheet__"].fillna("").astype(str).where(out["__sheet__"].astype(bool), source_sheet)
    return out


def merge_from_provenance(existing: Optional[str], added: str) -> str:
    parts = []
    if existing:
        parts.extend([p for p in str(existing).split("|") if p])
    parts.append(added)
    return "|".join(dict.fromkeys(parts))  # de-dup order preserved
