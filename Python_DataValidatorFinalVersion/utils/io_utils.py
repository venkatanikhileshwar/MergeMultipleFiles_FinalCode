from __future__ import annotations
from typing import Dict, List, Optional
import io
import os
import pandas as pd
from utils.normalize_utils import drop_completely_empty_rows

def detect_type(filename: str) -> str:
    if not filename:
        return "other"
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".xlsx", ".xls"):
        return "excel"
    if ext == ".csv":
        return "csv"
    if ext in (".txt", ".tsv", ".dat"):
        return "txt"
    if ext == ".pdf":
        return "pdf"
    return "other"

_DELIMS = [",", "|", "\t", ";", "~", "^"]

def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")

def _sniff_delimiter(text: str) -> Optional[str]:
    if not text:
        return None
    first = text.splitlines()[0] if text.splitlines() else text
    counts = {d: first.count(d) for d in _DELIMS}
    best = max(counts, key=counts.get)
    return best if counts.get(best, 0) > 0 else None

def _read_text_like(file) -> pd.DataFrame:
    try:
        file.seek(0)
    except Exception:
        pass
    raw = file.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="ignore")
    text = _decode_bytes(raw)
    sep = _sniff_delimiter(text)
    fh = io.StringIO(text)
    if sep is None:
        df = pd.read_csv(fh, dtype=object, sep=None, engine="python")
    else:
        df = pd.read_csv(fh, dtype=object, sep=sep, engine="python")
    return df

def load_headers(file, file_kind: str) -> Dict[str, List[str]]:
    if file_kind == "excel":
        xls = pd.ExcelFile(file)
        scopes = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, nrows=0, dtype=object)
            scopes[sheet] = list(df.columns)
        return scopes
    if file_kind in ("csv", "txt"):
        df = _read_text_like(file)
        df = drop_completely_empty_rows(df)
        return {"Data": list(df.columns)}
    return {"Data": []}

def read_sheet_df(file, file_kind: str, sheet_name: str = None) -> pd.DataFrame:
    if file_kind == "excel":
        df = pd.read_excel(file, sheet_name=sheet_name, dtype=object)
        return drop_completely_empty_rows(df)
    if file_kind in ("csv", "txt"):
        df = _read_text_like(file)
        return drop_completely_empty_rows(df)
    return pd.DataFrame()
