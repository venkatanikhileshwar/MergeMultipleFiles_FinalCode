from __future__ import annotations
from typing import List, Tuple
import re
import pandas as pd

_DEFAULT_BLANKS = ["", "-", "_", "NA", "N/A", "None", "null"]
_NUM_CLEAN_RE = re.compile(r"[,\s]")
_CURR_CHARS = "$€£¥₹"

_NEVER_DATE_DEFAULTS = (
    "code", "id", "key", "zip", "zipcode", "cbsa", "npi",
    "hcpcs", "icd", "upc", "sku", "ssn"
)

def _name_never_date(colname: str, cfg: dict) -> bool:
    pats = tuple(str(x).lower() for x in cfg.get("never_date_name_patterns", _NEVER_DATE_DEFAULTS))
    c = str(colname).lower()
    return any(p in c for p in pats)

def drop_completely_empty_rows(df: pd.DataFrame, blank_equivalents=None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    blanks = set((blank_equivalents or _DEFAULT_BLANKS))
    blanks = {str(x).lower() for x in blanks}
    df2 = df.dropna(how="all").copy()
    if df2.empty:
        return df2
    s = df2.astype(str).apply(lambda col: col.str.strip())
    s = s.mask(s.applymap(lambda x: str(x).lower() in blanks), "")
    row_empty = s.eq("").all(axis=1)
    out = df2.loc[~row_empty].copy()
    col_empty = s.eq("").all(axis=0)
    if col_empty.any():
        out = out.loc[:, ~col_empty].copy()
    return out

def _clean_number_like(s: pd.Series, cfg: dict) -> pd.Series:
    cs = "".join(cfg.get("currency_symbols", list(_CURR_CHARS)))
    s2 = s.str.replace("\u00A0", " ", regex=False).str.replace("\u200B", "", regex=False)
    s2 = s2.str.replace(f"[{re.escape(cs)}%]", "", regex=True)
    if cfg.get("strip_commas_and_spaces", True):
        s2 = s2.str.replace(_NUM_CLEAN_RE, "", regex=True)
    if cfg.get("remove_underscores_in_numbers", False):
        s2 = s2.str.replace("_", "", regex=False)
    if cfg.get("treat_parentheses_as_negative", True):
        s2 = s2.str.replace(r"^\((.+)\)$", r"-\1", regex=True)
    return s2

def _to_string(series: pd.Series, cfg: dict) -> pd.Series:
    s = series.astype("string")
    s = s.str.replace("\u00A0", " ", regex=False).str.replace("\u200B", "", regex=False)
    s = s.str.strip()
    if cfg.get("collapse_internal_spaces", True):
        s = s.str.replace(r"\s+", " ", regex=True)
    configured = {str(b).lower() for b in cfg.get("blank_equivalents", _DEFAULT_BLANKS)}
    lower = s.str.lower()
    dash_like = lower.str.match(r"^[-_]+$", na=False)
    null_like = lower.isin({"null", "(null)", "<null>", "none", "n/a", "na"})
    s = s.mask(lower.isin(configured) | dash_like | null_like, pd.NA)
    case = str(cfg.get("string_case", "lower")).lower()
    if case == "lower":
        s = s.str.casefold()
    elif case == "upper":
        s = s.str.upper()
    return s

def _parse_dates_to_dt(series: pd.Series) -> pd.Series:
    as_str = series.astype("string").str.strip()
    eight = as_str.str.match(r"^\d{8}$", na=False)
    if eight.any():
        as_str.loc[eight] = as_str.loc[eight].str.replace(
            r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True
        )
    parsed = pd.to_datetime(as_str, errors="coerce", infer_datetime_format=True)
    return parsed

def _apply_excel_serial_fallback(parsed: pd.Series, as_str: pd.Series) -> pd.Series:
    bad = parsed.isna() & as_str.str.match(r"^\d+$", na=False)
    if bad.any():
        ser = pd.to_numeric(as_str[bad], errors="coerce")
        parsed.loc[bad] = pd.to_datetime(ser, unit="D", origin="1899-12-30", errors="coerce")
    return parsed

def _detect_date_ratio(series: pd.Series, allow_serial: bool) -> Tuple[pd.Series, float]:
    s = series
    mask_input = s.notna() & (s.astype(str).str.strip() != "")
    parsed = _parse_dates_to_dt(s)
    if allow_serial:
        parsed = _apply_excel_serial_fallback(parsed, s.astype("string").str.strip())
    if not mask_input.any():
        return parsed, 0.0
    ratio = (parsed.notna() & mask_input).sum() / mask_input.sum()
    return parsed, float(ratio)

def _is_mostly_numeric(series: pd.Series, cfg: dict, thr: float) -> Tuple[pd.Series, float]:
    s = _to_string(series, cfg)
    s_clean = _clean_number_like(s.fillna(""), cfg)
    num = pd.to_numeric(s_clean, errors="coerce")
    mask_input = s.notna() & (s.astype(str).str.strip() != "")
    if not mask_input.any():
        return num, 0.0
    ratio = num.notna().sum() / mask_input.sum()
    return num, float(ratio)

def _name_hints_date(colname: str) -> bool:
    c = str(colname).lower()
    return ("date" in c) or c.endswith(("dt", "dat"))

def normalize_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    nc = cfg or {}
    fmt = nc.get("date_format", "%Y-%m-%d")
    num_thr = float(nc.get("numeric_majority_threshold", 0.8))
    date_thr = float(nc.get("date_detect_threshold", 0.6))
    out = drop_completely_empty_rows(df, nc.get("blank_equivalents")).copy()

    for col in out.columns:
        raw = out[col]
        s_str = _to_string(raw, nc)

        never_date = _name_never_date(col, nc)
        allow_serial = bool(nc.get("allow_excel_serial_detection", True)) and not never_date

        if not never_date:
            parsed_dates, date_ratio = _detect_date_ratio(raw, allow_serial)
        else:
            parsed_dates, date_ratio = (pd.to_datetime(pd.Series([pd.NaT]*len(s_str), index=s_str.index)), 0.0)

        num_series, num_ratio = _is_mostly_numeric(raw, nc, num_thr)
        has_leading_zero = s_str.dropna().str.match(r"^0\d+$").any()

        if (date_ratio >= date_thr) or (_name_hints_date(col) and date_ratio >= 0.4 and date_ratio >= num_ratio):
            out[col] = parsed_dates.dt.strftime(fmt)
        elif num_ratio >= num_thr and not has_leading_zero:
            if num_series.dropna().empty:
                out[col] = num_series
            elif (num_series.dropna() == num_series.dropna().astype(int)).all():
                out[col] = num_series.astype("Int64")
            else:
                out[col] = num_series.astype(float)
        else:
            out[col] = s_str
    return out

def harmonize_keys(file_df: pd.DataFrame, db_df: pd.DataFrame, keys: List[str], cfg: dict) -> None:
    if file_df is None or db_df is None or not keys:
        return
    fmt = (cfg or {}).get("date_format", "%Y-%m-%d")
    num_thr = float((cfg or {}).get("numeric_majority_threshold", 0.8))
    date_thr = float((cfg or {}).get("date_detect_threshold", 0.6))

    for k in keys:
        if k not in file_df.columns:
            continue
        never_date = _name_never_date(k, cfg)
        allow_serial = bool((cfg or {}).get("allow_excel_serial_detection", True)) and not never_date

        f_raw = file_df[k]
        d_raw = db_df[k] if k in db_df.columns else None

        f_str = _to_string(f_raw, cfg)
        d_str = _to_string(d_raw, cfg) if d_raw is not None else None

        has_leading_zero = f_str.dropna().str.match(r"^0\d+$").any()

        if not never_date:
            f_dates, f_date_ratio = _detect_date_ratio(f_raw, allow_serial)
        else:
            f_dates, f_date_ratio = (pd.to_datetime(pd.Series([pd.NaT]*len(f_str), index=f_str.index)), 0.0)

        if d_raw is not None and not never_date:
            d_dates, d_date_ratio = _detect_date_ratio(d_raw, allow_serial)
        else:
            d_dates, d_date_ratio = (pd.to_datetime(pd.Series([pd.NaT]*len(f_str), index=f_str.index)), 0.0)

        f_num, f_num_ratio = _is_mostly_numeric(f_str, cfg, num_thr)
        if d_str is not None:
            d_num, d_num_ratio = _is_mostly_numeric(d_str, cfg, num_thr)
        else:
            d_num, d_num_ratio = (pd.Series(pd.NA, index=f_str.index, dtype="Float64"), 0.0)

        if (not never_date) and ((f_date_ratio >= date_thr and d_date_ratio >= 0.4) or (d_date_ratio >= date_thr and f_date_ratio >= 0.4)):
            file_df[k] = f_dates.dt.strftime(fmt)
            if k in db_df.columns:
                db_df[k] = d_dates.dt.strftime(fmt)
        elif (f_num_ratio >= 0.7) and (d_num_ratio >= 0.7) and not has_leading_zero:
            file_df[k] = f_num
            if k in db_df.columns:
                db_df[k] = d_num
        else:
            file_df[k] = f_str
            if k in db_df.columns:
                db_df[k] = d_str
