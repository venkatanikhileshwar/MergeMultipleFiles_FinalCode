# utils/validate_utils.py
from __future__ import annotations

import re
import logging
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------
# Normalization helper functions
# ------------------------------

def _is_empty_equiv(val, blank_equivalents: List[str] | None) -> bool:
    """Treat configured blanks as empty."""
    if pd.isna(val):
        return True
    s = str(val).strip()
    if not blank_equivalents:
        return s == ""
    return s in set(blank_equivalents)


def _normalize_text(s: str, norm_cfg: dict) -> str:
    """Trim + optionally collapse internal whitespace."""
    s = s.strip()
    if norm_cfg.get("collapse_internal_spaces", False):
        s = re.sub(r"\s+", " ", s)
    return s


def _normalize_number(s: str, norm_cfg: dict):
    """
    Remove currency symbols (from config), thousands separators, spaces,
    treat parentheses as negatives if configured, and parse numeric.
    """
    txt = s

    # remove currency symbols from config (fallback to common)
    cfg_syms = norm_cfg.get("currency_symbols", ["$", "€", "£", "¥", "₹"])
    if cfg_syms:
        # Remove any of these characters
        txt = "".join(ch for ch in txt if ch not in set(cfg_syms))

    # (123) -> -123
    if norm_cfg.get("treat_parentheses_as_negative", False):
        if re.fullmatch(r"\(\s*[^()]+\s*\)", txt):
            txt = "-" + txt.strip()[1:-1]

    # Remove commas & spaces if configured
    if norm_cfg.get("strip_commas_and_spaces", True):
        txt = re.sub(r"[,\s]", "", txt)

    if norm_cfg.get("remove_underscores_in_numbers", False):
        txt = txt.replace("_", "")

    # Try numeric parse
    num = pd.to_numeric(txt, errors="coerce")
    return num


def _normalize_date(val, date_format: str):
    """Parse arbitrary input to datetime and reformat to configured format."""
    dt = pd.to_datetime(val, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.strftime(date_format)


def _norm_for_compare(val, col_name: str, norm_cfg: dict):
    """
    Normalize a single value for comparison and return a (type_tag, normalized_value) pair.
    type_tag ∈ {"empty", "number", "date", "text"}.
    """
    blanks = norm_cfg.get("blank_equivalents", [])
    if _is_empty_equiv(val, blanks):
        return ("empty", None)

    raw = str(val)
    raw = _normalize_text(raw, norm_cfg)

    # date columns (exact names, case-insensitive) -> normalized format
    date_cols = {c.lower() for c in (norm_cfg.get("date_columns") or [])}
    if col_name.lower() in date_cols and norm_cfg.get("date_format"):
        formatted = _normalize_date(raw, norm_cfg["date_format"])
        if formatted is not None:
            return ("date", formatted)

    # try numeric
    num = _normalize_number(raw, norm_cfg)
    if pd.notna(num):
        return ("number", float(num))

    # fallback text (lower-cased for case-insensitive match)
    return ("text", raw.lower())


# ------------------------------
# Key presence (Missing) checks
# ------------------------------

def _clean_key_series(s: pd.Series, norm_cfg: Optional[dict] = None) -> pd.Series:
    """
    Normalize key series to string for matching:
      - strip() whitespace
      - drop blank-equivalent keys
    """
    norm_cfg = norm_cfg or {}
    blanks = set(norm_cfg.get("blank_equivalents", []))
    out = s.astype("string").fillna("").str.strip()
    # blank-equivalent -> NaN
    if blanks:
        out = out.mask(out.isin(blanks), pd.NA)
    return out


def missing_in_db_by_key(
    file_df: pd.DataFrame, db_df: pd.DataFrame, key_col: str, norm_cfg: Optional[dict] = None
) -> pd.DataFrame:
    """
    Return the FULL rows from file_df whose keys are not present in db_df.
    Keys are matched after trimming and blank-normalization.
    """
    if file_df.empty or db_df.empty or key_col not in file_df.columns or key_col not in db_df.columns:
        return pd.DataFrame()

    f_keys = _clean_key_series(file_df[key_col], norm_cfg)
    d_keys = _clean_key_series(db_df[key_col], norm_cfg)

    keep_mask = f_keys.notna() & ~f_keys.isin(set(d_keys.dropna().astype(str)))
    missing_rows = file_df.loc[keep_mask].copy()
    return missing_rows.reset_index(drop=True)


def missing_in_file_by_key(
    file_df: pd.DataFrame, db_df: pd.DataFrame, key_col: str, norm_cfg: Optional[dict] = None
) -> pd.DataFrame:
    """
    Return the FULL rows from db_df whose keys are not present in file_df.
    Keys are matched after trimming and blank-normalization.
    """
    if file_df.empty or db_df.empty or key_col not in file_df.columns or key_col not in db_df.columns:
        return pd.DataFrame()

    f_keys = _clean_key_series(file_df[key_col], norm_cfg)
    d_keys = _clean_key_series(db_df[key_col], norm_cfg)

    keep_mask = d_keys.notna() & ~d_keys.isin(set(f_keys.dropna().astype(str)))
    missing_rows = db_df.loc[keep_mask].copy()
    return missing_rows.reset_index(drop=True)


# ------------------------------
# Null & Type checks
# ------------------------------

def null_checks(
    df: pd.DataFrame,
    columns: List[str],
    norm_cfg: Optional[dict] = None
) -> pd.DataFrame:
    """
    Return full rows that contain at least one null/blank-equivalent value in the given columns,
    and annotate which columns are null in each row.
    """
    if df.empty or not columns:
        return pd.DataFrame()

    norm_cfg = norm_cfg or {}
    blanks = set(norm_cfg.get("blank_equivalents", []))

    # Build a mask dataframe marking empty equivalence per column
    null_mask = pd.DataFrame(False, index=df.index, columns=columns)
    for c in columns:
        s = df[c]
        # treat blank equivalents & actual NaN as null
        col_is_null = s.isna() | s.astype(str).str.strip().isin(blanks)
        null_mask[c] = col_is_null

    rows_with_nulls = null_mask.any(axis=1)
    if not rows_with_nulls.any():
        return pd.DataFrame()

    result_rows = []
    for idx in df[rows_with_nulls].index:
        row_data = df.loc[idx, columns].copy()
        # format nulls for display
        null_cols = [c for c in columns if null_mask.loc[idx, c]]
        display_row = df.loc[idx, :].copy()
        for c in null_cols:
            display_row[c] = "<NULL>"
        # add metadata
        row_dict = display_row.to_dict()
        row_dict["null_columns"] = null_cols
        row_dict["null_count"] = len(null_cols)
        result_rows.append(row_dict)

    return pd.DataFrame(result_rows)


def type_checks(
    df: pd.DataFrame,
    expected_spec: Dict[str, str],
    norm_cfg: Optional[dict] = None
) -> pd.DataFrame:
    """
    Simple type validation based on a spec:
      expected_spec = {column: 'date'|'number'}
    Returns rows: [row_index, column, value, expected_type, issue]
    """
    if df.empty or not expected_spec:
        return pd.DataFrame()

    issues = []
    for col, expected in expected_spec.items():
        if col not in df.columns:
            continue

        if expected == "date":
            invalid = df[col].notna() & pd.to_datetime(df[col], errors="coerce").isna()
            for i in df[invalid].index:
                issues.append({
                    "row_index": i,
                    "column": col,
                    "value": df.at[i, col],
                    "expected_type": "date",
                    "issue": "invalid_date",
                })
        elif expected == "number":
            invalid = df[col].notna() & pd.to_numeric(df[col], errors="coerce").isna()
            for i in df[invalid].index:
                issues.append({
                    "row_index": i,
                    "column": col,
                    "value": df.at[i, col],
                    "expected_type": "number",
                    "issue": "invalid_number",
                })

    return pd.DataFrame(issues)


# ------------------------------
# Value mismatches by KEY
# ------------------------------

def compute_mismatches_by_key(
    file_df: pd.DataFrame,
    db_df: pd.DataFrame,
    key_col: str,
    norm_cfg: Optional[dict] = None
) -> pd.DataFrame:
    """
    For keys present on both sides, compare non-key columns.
    Uses robust normalization guided by norm_cfg:
      - case-insensitive column matching (excluding key)
      - trims + (optional) collapse spaces
      - removes currency symbols from config; strips commas/spaces; parentheses→negative (opt)
      - date columns compared using configured date_format
      - blank equivalents respected
    Returns rows: [key, column, file_value, db_value, mismatch_type]
    """
    norm_cfg = norm_cfg or {}

    # Key presence
    if key_col not in file_df.columns or key_col not in db_df.columns:
        return pd.DataFrame()

    # Case-insensitive alignment of non-key columns
    f_map = {c.lower(): c for c in file_df.columns if c != key_col}
    d_map = {c.lower(): c for c in db_df.columns if c != key_col}
    common_lower = list(set(f_map.keys()) & set(d_map.keys()))
    if not common_lower:
        return pd.DataFrame()

    # Unique-by-key on each side
    f_unique = file_df.drop_duplicates(subset=[key_col], keep="first")
    d_unique = db_df.drop_duplicates(subset=[key_col], keep="first")

    # Inner join on key (only rows with the key on both sides can mismatch)
    merged = f_unique.merge(d_unique, on=key_col, how="inner", suffixes=("_file", "_db"))
    if merged.empty:
        return pd.DataFrame()

    out_rows = []
    for c_low in common_lower:
        cf = f"{f_map[c_low]}_file"
        cd = f"{d_map[c_low]}_db"
        if cf not in merged.columns or cd not in merged.columns:
            continue

        subset = merged[[key_col, cf, cd]]
        for _, row in subset.iterrows():
            key_val = row[key_col]
            fval = row[cf]
            dval = row[cd]

            # Normalize both sides using config
            t1, n1 = _norm_for_compare(fval, f_map[c_low], norm_cfg)
            t2, n2 = _norm_for_compare(dval, d_map[c_low], norm_cfg)

            # both empty -> equal
            if t1 == "empty" and t2 == "empty":
                continue

            # one empty -> mismatch
            if (t1 == "empty") ^ (t2 == "empty"):
                out_rows.append({
                    "key": key_val,
                    "column": f_map[c_low],  # original file column name
                    "file_value": fval,
                    "db_value": dval,
                    "mismatch_type": "file_empty" if t1 == "empty" else "db_empty",
                })
                continue

            # same type & equal normalized value -> OK
            equal = (t1 == t2) and (n1 == n2)
            if not equal:
                out_rows.append({
                    "key": key_val,
                    "column": f_map[c_low],
                    "file_value": fval,
                    "db_value": dval,
                    "mismatch_type": "value_different" if t1 == t2 else f"type_diff({t1}->{t2})",
                })

    return pd.DataFrame(out_rows)
