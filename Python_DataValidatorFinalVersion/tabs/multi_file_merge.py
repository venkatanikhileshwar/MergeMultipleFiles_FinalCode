# tabs/multi_file_merge.py
# -*- coding: utf-8 -*-
"""Multi-File → DB (Append / Union)"""

from __future__ import annotations

# --- IMPORT BOOTSTRAP: add PROJECT ROOT ONLY ---
import os, sys, io, json
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
import sqlalchemy
from collections import defaultdict, OrderedDict

# Optional project IO helpers
try:
    import utils.io_utils as iox
except Exception:
    iox = None

# Multi utils you already have
from utils.Multi.header_utils import normalize_headers
from utils.Multi.key_suggest_utils import find_candidate_keys, score_keys, recommend_default_key
from utils.Multi.provenance_utils import add_provenance_cols
from utils.Multi.merge_plan_utils import MergePlan

# Reuse your DB utilities and validators (from File→DB flow)
from utils.sql_utils import get_engine, run_query_preview, run_query_full
from utils.validate_utils import compute_mismatches_by_key

# Define PROVENANCE_COLS and is_real_column here since they're missing from imports
PROVENANCE_COLS = {"__source__", "__sheet__"}

def is_real_column(col_name: str) -> bool:
    """Check if column is a real data column (not provenance)"""
    return str(col_name) not in PROVENANCE_COLS

# ============================
# Models & helpers
# ============================
@dataclass
class UploadedPart:
    label: str            # e.g., "file.xlsx / Sheet1" or "file.csv"
    source_file: str
    source_sheet: Optional[str]
    df: pd.DataFrame


def _drop_all_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.replace({"nan": "", "NaN": "", "None": "", "NULL": ""})
    mask_all_blank = tmp.applymap(lambda x: isinstance(x, str) and x.strip() == "").all(axis=1)
    return df.loc[~mask_all_blank].copy()


def _normalize_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Headers normalized; all cells as trimmed strings; drop all-blank rows."""
    norm_cols, mp = normalize_headers(list(df.columns))
    out = df.copy()
    out.columns = norm_cols
    if not out.empty:
        for c in out.columns:
            out[c] = out[c].astype(str).map(lambda x: x.strip())
            out[c] = out[c].replace({"nan": "", "NaN": "", "None": "", "NULL": ""})
    out = _drop_all_blank_rows(out)
    return out, mp


def _read_uploaded_file(uploaded, preferred_sheet: Optional[str] = None) -> List[UploadedPart]:
    """Reliable reader (Excel/CSV/TXT). PDF not supported here."""
    name = uploaded.name
    parts: List[UploadedPart] = []

    # bytes once
    try:
        file_bytes = uploaded.getvalue()
    except Exception:
        uploaded.seek(0)
        file_bytes = uploaded.read()
    bbuf = io.BytesIO(file_bytes)

    ext = os.path.splitext(name)[1].lower()
    ftype = None
    if iox is not None:
        try:
            ftype = iox.detect_type(name)
        except Exception:
            ftype = None

    # Excel
    if ftype == "excel" or ext in (".xlsx", ".xls"):
        try:
            xl = pd.ExcelFile(bbuf)
            sheet = preferred_sheet or (xl.sheet_names[0] if xl.sheet_names else None)
            df = xl.parse(sheet_name=sheet, dtype=str)
            parts.append(UploadedPart(label=f"{name} / {sheet or 'Sheet1'}",
                                      source_file=name, source_sheet=sheet or "Sheet1", df=df))
            return parts
        except Exception as e:
            st.error(f"Failed to read {name} as Excel: {e}")
            return []

    # CSV/TXT
    if ftype in {"csv", "txt"} or ext in (".csv", ".txt"):
        sample = file_bytes[:4096].decode("utf-8", errors="ignore")
        delim = "\t" if "\t" in sample else "|" if "|" in sample else ";" if ";" in sample else ","
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), delimiter=delim, dtype=str, engine="python")
            parts.append(UploadedPart(label=name, source_file=name, source_sheet="-", df=df))
            return parts
        except Exception as e:
            st.error(f"Failed to read {name} as text: {e}")
            return []

    st.warning(f"{name}: unsupported type for this tab.")
    return []

def _format_number_like_as_str(s: pd.Series) -> pd.Series:
    """If value is float and integer-like, show without .0; else string."""
    def _fmt(v):
        try:
            # Treat 12.0 as "12" for display
            if isinstance(v, float) and float(v).is_integer():
                return str(int(v))
        except Exception:
            pass
        return "" if v is None else str(v)
    return s.map(_fmt)


def _df_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=(name[:31] or "Sheet1"), index=False)
    buf.seek(0)
    return buf.read()


def _to_null_factory(blank_equivs):
    lower_equivs = {str(x).lower() for x in blank_equivs}
    def to_null(v):
        if pd.isna(v):
            return pd.NA
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() in lower_equivs:
                return pd.NA
        return v
    return to_null


def _to_key_str(s: pd.Series) -> pd.Series:
    """Comparable string key (fixes float/object mismatch)."""
    def conv(v):
        if pd.isna(v):
            return pd.NA
        if isinstance(v, float):
            if v.is_integer():
                return str(int(v))
            vstr = str(v)
            return vstr.rstrip("0").rstrip(".") if "." in vstr else vstr
        return str(v).strip()
    return s.map(conv).astype("string")


# =============== Merge engines (Append / Union) ===============
def _concat_schema_union(parts: List[UploadedPart]) -> pd.DataFrame:
    """Append mode: union schema + provenance columns."""
    frames: List[pd.DataFrame] = []
    for p in parts:
        df = add_provenance_cols(p.df.copy(), p.source_file, p.source_sheet or "-")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _outer_union_reduce(
    parts: List[UploadedPart],
    key_cols: List[str],
    anchor_label: str,
    conflict_log: List[pd.DataFrame],
) -> pd.DataFrame:
    """
    FULL OUTER JOIN semantics across N files with anchor/first-non-blank winning rule.
    Differences logged to conflict_log (list of DataFrames).
    """
    # choose anchor
    label_to_df = {p.label: p.df for p in parts}
    working = label_to_df[anchor_label].copy()

    plan = MergePlan()
    plan.start("join", key_cols, "OUTER / UNION")

    # sequentially union other files
    for p in [q for q in parts if q.label != anchor_label]:
        right = p.df
        missing = [k for k in key_cols if k not in right.columns]
        if missing:
            plan.add_step(f"⚠ Skipped {p.label} (missing key columns: {', '.join(missing)})")
            continue
        merged = pd.merge(
            working, right, on=key_cols, how="outer", suffixes=("", "__r")
        )

        # compute conflicts: shared columns where both sides non-blank and different
        conflicts_rows = []
        for col in right.columns:
            if col in key_cols:
                continue
            rcol = f"{col}__r"
            if rcol not in merged.columns:
                # brand new column, nothing to compare yet
                continue

            # both present → compare (treat "", "nan", "None", "NULL" as blank)
            a = merged[col].astype(str).str.strip()
            b = merged[rcol].astype(str).str.strip()
            def _blank(s: pd.Series) -> pd.Series:
                return (s == "") | (s.str.lower().isin({"nan","none","null"}))
            mask_conflict = (~_blank(a)) & (~_blank(b)) & (a != b)
            if mask_conflict.any():
                cdf = merged.loc[mask_conflict, key_cols].copy()
                cdf["column"] = col
                cdf["anchor_value"] = a[mask_conflict].values
                cdf["incoming_value"] = b[mask_conflict].values
                cdf["__source__"] = p.source_file
                conflicts_rows.append(cdf)

            # anchor/first-non-blank wins (fill blanks from right, else keep anchor)
            mask_a_blank = _blank(a)
            merged.loc[mask_a_blank, col] = merged.loc[mask_a_blank, rcol]

            # drop right-suffixed column
            merged = merged.drop(columns=[rcol])

        if conflicts_rows:
            conflict_log.append(pd.concat(conflicts_rows, ignore_index=True))

        working = merged.reset_index(drop=True)
        plan.add_step(f"⚡ OUTER with {p.label} — rows now: {len(working):,}")

    # record plan text into session (optional for debugging/download later)
    st.session_state.setdefault("multi_merge_plan_text", plan.to_text())
    return working


# ============================
# Streamlit UI
# ============================
def render():
    st.header("Multi-File → DB (Append / Union)")
    st.caption("Upload Excel/CSV/TXT. Choose **Append** or **Union on key**. PDFs are not supported here.")
    
    # --- persist union result across reruns ---
    if "union_result" not in st.session_state:
        st.session_state["union_result"] = None
    if "union_conflicts" not in st.session_state:
        st.session_state["union_conflicts"] = None

    uploads = st.file_uploader(
        "Upload one or more files",
        type=["xlsx", "xls", "csv", "txt"],
        accept_multiple_files=True,
    )
    if not uploads:
        st.info("Upload at least one file to begin.")
        return

    # Read & normalize each uploaded file
    parts: List[UploadedPart] = []
    file_to_map: Dict[str, dict] = {}   # label -> {original->normalized}
    file_to_df: Dict[str, pd.DataFrame] = {}  # label -> normalized df

    for up in uploads:
        for p in _read_uploaded_file(up):
            norm_df, mp = _normalize_df(p.df)
            parts.append(UploadedPart(label=p.label, source_file=p.source_file, source_sheet=p.source_sheet, df=norm_df))
            file_to_map[p.label] = mp
            file_to_df[p.label] = norm_df

    # Files summary + header mapping
    with st.expander("Files summary / Column mapping", expanded=True):
        fs_rows = [[p.source_file, (p.source_sheet or "-"), len(p.df), len(p.df.columns)] for p in parts]
        st.dataframe(pd.DataFrame(fs_rows, columns=["Source File", "Sheet", "Rows", "Cols"]), use_container_width=True)

        st.markdown("**Header mapping (original → normalized)**")
        map_rows = []
        for label, mp in file_to_map.items():
            for original, normalized in mp.items():
                map_rows.append({"file": label, "original": original, "normalized": normalized})
        st.dataframe(pd.DataFrame(map_rows, columns=["file", "original", "normalized"]), use_container_width=True)

    if not parts:
        st.warning("No readable tables.")
        return

    # Choose mode
    mode = st.radio("Merge method", ["Append rows (stack)", "Union on key (FULL OUTER)"], horizontal=False)

    # ================= Append =================
    if mode.startswith("Append"):
        merged = _concat_schema_union(parts)

        st.subheader("Append preview")
        st.write(f"Total rows across files: {len(merged):,}")
        st.dataframe(merged.head(30), use_container_width=True)

        rm_dups = st.checkbox("Remove exact duplicate rows (full-row equality)", value=True)
        if rm_dups and not merged.empty:
            before = len(merged)
            merged = merged.drop_duplicates()
            st.info(f"Exact duplicates removed: {before - len(merged):,}")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="Download merged (CSV)",
                data=merged.to_csv(index=False).encode("utf-8"),
                file_name="Merged_Append.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                label="Download merged (Excel)",
                data=_df_to_excel_bytes({"Merged_Append": merged}),
                file_name="Merged_Append.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # Inline DB compare (optional)
        _db_compare_ui(merged)
        return

    # ================= Union on key (FULL OUTER) =================
    st.subheader("Union settings")

    # Suggest key(s) (ID-first ranking)
    filtered_for_keys = {
        label: df.loc[:, [c for c in df.columns if is_real_column(c)]]
        for label, df in file_to_df.items()
    }
    scored = score_keys(filtered_for_keys, find_candidate_keys(filtered_for_keys))
    if not scored:
        st.error("Could not suggest any keys. Make sure files share at least one column.")
        return

    key_options = [
        " + ".join(s["key"]) + (
            f"  •  {'ID' if s['identifier'] else 'not recommended'}"
            f"  •  coverage {s['coverage_pct']}%  •  dups {s['dup_counts']}"
        ) for s in scored
    ]
    default_key_txt = " + ".join(recommend_default_key(scored)) if scored else ""
    default_idx = 0
    for i, o in enumerate(key_options):
        if default_key_txt and o.startswith(default_key_txt):
            default_idx = i; break

    key_pick = st.selectbox("Select PRIMARY KEY for Union", options=key_options, index=default_idx)
    key_cols = key_pick.split("  •  ")[0].split(" + ") if key_pick else []

    # anchor = largest table that has the key
    files_with_key = [p for p in parts if all(k in p.df.columns for k in key_cols)]
    if not files_with_key:
        st.error("No uploaded file contains all selected key columns.")
        return
    anchor_part = max(files_with_key, key=lambda p: len(p.df))

    st.caption(f"Anchor = {anchor_part.label} (largest table with the key)")

    # Run union (compute once and persist)
    if st.button("Run UNION on key", type="primary", use_container_width=True):
        conflicts_all: List[pd.DataFrame] = []
        unioned = _outer_union_reduce(parts, key_cols, anchor_part.label, conflicts_all)

        conflicts = (
            pd.concat(conflicts_all, ignore_index=True)
            if conflicts_all else
            pd.DataFrame(columns=key_cols + ["column", "anchor_value", "incoming_value", "__source__"])
        )

        # persist to session so later button clicks (e.g., Test Connection) don't clear it
        st.session_state["union_result"] = unioned
        st.session_state["union_conflicts"] = conflicts

    # Display union results if available
    if st.session_state.get("union_result") is not None:
        unioned = st.session_state["union_result"]
        conflicts = st.session_state["union_conflicts"]
        
        st.success(f"Union complete. Rows: {len(unioned):,} • Conflicts: {len(conflicts):,}")

        # Show merge plan if available
        if "multi_merge_plan_text" in st.session_state:
            with st.expander("Merge Plan", expanded=True):
                st.text(st.session_state["multi_merge_plan_text"])

        st.subheader("Union Result")
        st.dataframe(unioned.head(30), use_container_width=True)

        if not conflicts.empty:
            st.subheader("Conflicts")
            st.dataframe(conflicts.head(30), use_container_width=True)

        # Downloads for union results
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "Download Union CSV",
                data=unioned.to_csv(index=False).encode("utf-8"),
                file_name="Union_Result.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            if not conflicts.empty:
                st.download_button(
                    "Download Conflicts CSV",
                    data=conflicts.to_csv(index=False).encode("utf-8"),
                    file_name="Union_Conflicts.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.write("No conflicts to download")
        with c3:
            # Excel bundle
            try:
                xlsx_buffer = io.BytesIO()
                with pd.ExcelWriter(xlsx_buffer, engine='xlsxwriter') as writer:
                    unioned.to_excel(writer, sheet_name='Union_Result', index=False)
                    if not conflicts.empty:
                        conflicts.to_excel(writer, sheet_name='Conflicts', index=False)
                xlsx_buffer.seek(0)
                
                st.download_button(
                    "Download Excel Bundle",
                    data=xlsx_buffer.getvalue(),
                    file_name="Union_Bundle.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Excel creation failed: {e}")

        # Inline DB compare for union result
        _db_compare_ui(unioned)


# =============== Inline File→DB compare (borrows your File→DB flow) ===============
def _load_cfg():
    # same logic as file_to_db: first try config/appconfig.json, then app root
    try:
        with open(os.path.join(PROJECT_ROOT, "config", "appconfig.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        with open(os.path.join(PROJECT_ROOT, "appconfig.json"), "r") as f:
            return json.load(f)


def _default_match_index(db_cols, file_col):
    if not db_cols:
        return 0
    for i, c in enumerate(db_cols):
        if str(c).lower() == str(file_col).lower():
            return i + 1  # because option[0] is "— Ignore —"
    return 0


def _db_compare_ui(merged_df: pd.DataFrame):
    st.divider()
    st.subheader("Compare merged output with DB")

    if merged_df is None or merged_df.empty:
        st.info("Merged table is empty.")
        return

    cfg = _load_cfg()
    db_options = cfg["defaults"]["db"]["dropdown"]
    default_db = cfg["defaults"]["db"]["default"]
    connections = cfg["connections"]
    blank_equivs = set(cfg.get("normalization", {}).get("blank_equivalents", []))
    to_null = _to_null_factory(blank_equivs)

    # DB connect - use unique key to prevent interference with union results
    sel_db = st.selectbox("Database", db_options, index=db_options.index(default_db), key="db_compare_select")
    engine = get_engine(connections[sel_db]["url"])
    
    if st.button("Test Connection", key="mfdb_test_conn"):
        try:
            with engine.connect() as c:
                _ = c.execute(sqlalchemy.text("SELECT 1"))
            st.success("Connection OK")
        except Exception as e:
            st.error(f"Connection failed: {e}")

    # SQL
    sql = st.text_area("Enter SQL (SELECT ...)", height=110, key="mfdb_sql")
    if st.button("Preview Columns", key="mfdb_prev_cols"):
        try:
            cols = run_query_preview(engine, sql, limit=1000)
            if cols:
                st.success("Columns: " + ", ".join(cols))
                st.session_state["mfdb_cols"] = cols
            else:
                st.warning("No columns returned.")
                st.session_state.pop("mfdb_cols", None)
        except Exception as e:
            st.error(f"Preview failed: {e}")
    db_cols = st.session_state.get("mfdb_cols", [])

    # Mapping
    mapping = {}
    key_col = None
    if db_cols:
        st.markdown("**Column Mapping (Merged → DB)**")
        # Key selector
        key_candidates = [c for c in db_cols if str(c).lower() in ("record_id","id","patient_id","mrn","encounter_id")]
        default_key = key_candidates[0] if key_candidates else db_cols[0]
        key_col = st.selectbox("Select KEY column (DB)", db_cols, index=db_cols.index(default_key), key="db_key_select")

        options = ["— Ignore —"] + db_cols
        cols2 = st.columns(2)
        real_cols = [c for c in merged_df.columns if is_real_column(c)]
        for i, col in enumerate(real_cols):
            with cols2[i % 2]:
                idx = _default_match_index(db_cols, col)
                mapped = st.selectbox(f"📄 {col}", options, index=idx, key=f"mfdb_map_{i}")
                if mapped != "— Ignore —":
                    mapping[col] = mapped

    run_ready = (
        bool(db_cols)
        and bool(str(sql or "").strip())
        and key_col is not None
        and any(st.session_state.get(f"mfdb_map_{i}") == key_col for i in range(len([c for c in merged_df.columns if is_real_column(c)])))
    )
    
    if st.button("Run Validation", type="primary", use_container_width=True, disabled=not run_ready, key="run_validation_btn"):
        if not run_ready:
            st.error("⚠️ Ensure: preview DB cols • enter SQL • select DB key • map the key")
            return

        try:
            db_df = run_query_full(engine, sql)
            if db_df is None or db_df.empty:
                st.warning("DB query returned no rows.")
                return

            # Build mapped DF with first-non-null coalescing if same target mapped multi-times
            tgt_to_sources = OrderedDict()
            for col in merged_df.columns:
                tgt = mapping.get(col)
                if tgt:
                    tgt_to_sources.setdefault(tgt, []).append(col)

            merged_mapped_cols = {}
            for tgt, srcs in tgt_to_sources.items():
                if len(srcs) == 1:
                    ser = merged_df[srcs[0]]
                else:
                    tmp = merged_df[srcs].copy()
                    tmp = tmp.applymap(to_null)
                    ser = tmp.bfill(axis=1).iloc[:, 0]
                merged_mapped_cols[tgt] = ser

            file_mapped = pd.DataFrame(merged_mapped_cols)

            common_cols = [c for c in file_mapped.columns if c in db_df.columns]
            if key_col not in common_cols:
                st.error(f"KEY '{key_col}' must be mapped.")
                return

            file_sub = file_mapped[common_cols].copy()
            db_sub   = db_df[common_cols].copy()

            # Normalize non-key; coerce key to comparable string
            for c in [x for x in common_cols if x != key_col]:
                file_sub[c] = file_sub[c].map(to_null)
                db_sub[c]   = db_sub[c].map(to_null)
            file_sub[key_col] = _to_key_str(file_sub[key_col])
            db_sub[key_col]   = _to_key_str(db_sub[key_col])
            chk_cols = [c for c in common_cols if c != key_col]
            for c in chk_cols:
                file_sub[c] = _format_number_like_as_str(file_sub[c])
                db_sub[c]   = _format_number_like_as_str(db_sub[c])

            # Metrics
            m1, m2 = st.columns(2)
            m1.metric("Merged rows", f"{len(file_sub):,}")
            m2.metric("DB rows",     f"{len(db_sub):,}")

            results = {}

            # Missing keys
            keys_file = set(file_sub[key_col].dropna())
            keys_db   = set(db_sub[key_col].dropna())
            missing_in_db   = file_sub[file_sub[key_col].isin(keys_file - keys_db)]
            missing_in_file = db_sub[db_sub[key_col].isin(keys_db - keys_file)]
            if not missing_in_db.empty:
                with st.expander(f"Missing in DB ({len(missing_in_db)})"):
                    st.dataframe(missing_in_db, use_container_width=True, height=240)
                results["Missing_in_DB"] = missing_in_db
            if not missing_in_file.empty:
                with st.expander(f"Missing in File ({len(missing_in_file)})"):
                    st.dataframe(missing_in_file, use_container_width=True, height=240)
                results["Missing_in_File"] = missing_in_file

            # Duplicates / Matches (same as File→DB)
            SENTINEL = "__NULL__"
            f_cmp = file_sub.fillna(SENTINEL).astype(str)
            d_cmp = db_sub.fillna(SENTINEL).astype(str)

            exact_matches = pd.merge(
                f_cmp.drop_duplicates(),
                d_cmp.drop_duplicates(),
                on=common_cols, how="inner"
            )
            if not exact_matches.empty:
                with st.expander(f"Exact Matches ({len(exact_matches)})", expanded=False):
                    st.dataframe(exact_matches, use_container_width=True, height=240)
                results["Exact_Matches"] = exact_matches

            both = pd.concat([f_cmp.assign(_side="File"),
                              d_cmp.assign(_side="DB")], ignore_index=True)
            counts = (both.groupby(common_cols)["_side"]
                      .value_counts().unstack(fill_value=0).reset_index())
            if "File" not in counts.columns: counts["File"] = 0
            if "DB"  not in counts.columns: counts["DB"]  = 0
            counts["extras"] = counts["File"].sub(1).clip(lower=0) + counts["DB"].sub(1).clip(lower=0)
            extras = counts[counts["extras"] > 0]
            if not extras.empty:
                with st.expander(f"Extra Duplicates ({len(extras)})", expanded=True):
                    st.dataframe(extras, use_container_width=True, height=280)
                results["Duplicates"] = extras

            # Null Issues
            chk_cols = [c for c in common_cols if c != key_col]
            f_nulls = file_sub[file_sub[chk_cols].isna().any(axis=1)]
            d_nulls = db_sub[db_sub[chk_cols].isna().any(axis=1)]
            if not f_nulls.empty:
                with st.expander(f"File Null Issues ({len(f_nulls)})"):
                    st.dataframe(f_nulls, use_container_width=True, height=220)
                results["File_Null_Issues"] = f_nulls
            if not d_nulls.empty:
                with st.expander(f"DB Null Issues ({len(d_nulls)})"):
                    st.dataframe(d_nulls, use_container_width=True, height=220)
                results["DB_Null_Issues"] = d_nulls

            # Value Mismatches (by KEY)
            file_dedup = file_sub.drop_duplicates(subset=[key_col])
            db_dedup   = db_sub.drop_duplicates(subset=[key_col])
            mismatch_df = compute_mismatches_by_key(file_dedup, db_dedup, key_col)
            if not mismatch_df.empty:
                with st.expander("Value Mismatches", expanded=True):
                    st.dataframe(mismatch_df, use_container_width=True, height=300)
                results["Value_Mismatches"] = mismatch_df

            # Report bundle (Excel)
            results["Summary"] = pd.DataFrame([
                {"metric": "Merged_rows", "value": len(file_sub)},
                {"metric": "DB_rows",     "value": len(db_sub)},
                {"metric": "Key",         "value": key_col},
            ])
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                for name, df in results.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df.to_excel(writer, index=False, sheet_name=str(name)[:31])
            st.download_button(
                "📥 Download Validation Report",
                data=buf.getvalue(),
                file_name="Merged_to_DB_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Merged → DB validation failed: {e}")


if __name__ == "__main__":
    render()