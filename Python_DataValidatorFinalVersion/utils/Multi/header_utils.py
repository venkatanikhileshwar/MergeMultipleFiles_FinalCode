# ================================================
# filepath: utils/Multi/header_utils.py
# ================================================
from __future__ import annotations
import re
import pandas as pd
from typing import Dict, List, Tuple


__all__ = [
    "normalize_headers",
    "apply_header_map",
    "build_column_mapping_view",
]

# --- provenance helpers (Fix 1) ---
PROVENANCE_COLS = {"__source__", "__sheet__"}

def is_real_column(col: str) -> bool:
    """Hide provenance columns from any UI selector or key logic."""
    return str(col) not in PROVENANCE_COLS


_ws_re = re.compile(r"\s+")
_multi_underscore = re.compile(r"_+")

def _norm_col(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = _ws_re.sub("_", s)
    s = _multi_underscore.sub("_", s)
    return s

def normalize_headers(cols: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Return normalized headers and a map original->normalized.
    Rules: lowercase, trim, collapse spaces to `_`, collapse multiple `_`.
    """
    norm = []
    m: Dict[str, str] = {}
    for c in cols:
        nc = _norm_col(c)
        norm.append(nc)
        m[str(c)] = nc
    return norm, m


def apply_header_map(df: pd.DataFrame, orig_to_norm: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    # Avoid duplicate target names by de-duplicating with suffixes if needed
    target_counts: Dict[str, int] = {}
    new_cols: Dict[str, str] = {}
    for c in out.columns:
        tgt = orig_to_norm.get(str(c), str(c))
        if tgt in target_counts:
            target_counts[tgt] += 1
            tgt = f"{tgt}__{target_counts[tgt]}"
        else:
            target_counts[tgt] = 1
        new_cols[c] = tgt
    out = out.rename(columns=new_cols)
    return out


def build_column_mapping_view(files: Dict[str, pd.DataFrame], maps: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    rows = []
    for fname, df in files.items():
        mp = maps.get(fname, {})
        for oc in df.columns:
            rows.append({
                "file": fname,
                "original": str(oc),
                "normalized": mp.get(str(oc), str(oc))
            })
    return pd.DataFrame(rows, columns=["file", "original", "normalized"]) 
