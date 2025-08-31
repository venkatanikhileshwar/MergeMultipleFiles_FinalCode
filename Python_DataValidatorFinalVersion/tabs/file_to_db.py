# tabs/file_to_db.py
import streamlit as st
import pandas as pd
import json
import io
import sqlalchemy
import os
from collections import defaultdict, OrderedDict

from utils.io_utils import detect_type, read_sheet_df
from utils.pdf_utils import convert_uploaded_pdf_to_csv_temp
from utils.sql_utils import get_engine, run_query_preview, run_query_full
from utils.validate_utils import compute_mismatches_by_key


def get_unique_key(base_key: str) -> str:
    return f"file_db_{base_key}"


def _load_cfg():
    try:
        with open("config/appconfig.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        with open("appconfig.json", "r") as f:
            return json.load(f)


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


def _default_match_index(db_cols, file_col):
    if not db_cols:
        return 0
    for i, c in enumerate(db_cols):
        if str(c).lower() == str(file_col).lower():
            return i + 1  # because option[0] is "— Ignore —"
    return 0


def _make_unique_columns(cols) -> list:
    """Return unique column names (suffix dupes as name__1, name__2, ...)."""
    seen = defaultdict(int)
    out = []
    for c in ["" if c is None else str(c) for c in cols]:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def _merge_excel_sheets_to_df(dfs_by_sheet: dict) -> pd.DataFrame:
    """
    Merge {sheet_name: DataFrame} into one DataFrame.
    - Ensures unique headers per sheet
    - Adds 'source_sheet'
    - Aligns to union of columns
    - Skips empty sheets
    """
    items = []
    all_cols = set()

    normalized = {}
    for name, df in dfs_by_sheet.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        dfx = df.copy()
        dfx.columns = _make_unique_columns(dfx.columns)
        normalized[name] = dfx
        all_cols.update(dfx.columns.tolist())

    all_cols = list(all_cols)
    for name, df in normalized.items():
        aligned = df.reindex(columns=all_cols)
        aligned.insert(0, "source_sheet", name)
        items.append(aligned)

    return pd.concat(items, ignore_index=True) if items else pd.DataFrame(columns=["source_sheet"] + all_cols)


def render():
    cfg = _load_cfg()
    db_options = cfg["defaults"]["db"]["dropdown"]
    default_db = cfg["defaults"]["db"]["default"]
    connections = cfg["connections"]
    blank_equivs = set(cfg.get("normalization", {}).get("blank_equivalents", []))
    to_null = _to_null_factory(blank_equivs)

    st.header("File → DB")

    # ---------- File Upload ----------
    up = st.file_uploader(
        "Upload file (Excel/CSV/TXT/PDF)",
        type=["xlsx", "xls", "csv", "txt", "pdf"],
        key=get_unique_key("uploader")
    )

    file_df = None
    if up:
        kind = detect_type(up.name)
        try:
            if kind == "pdf":
                csv_tmp = convert_uploaded_pdf_to_csv_temp(up)
                if csv_tmp and os.path.exists(csv_tmp):
                    file_df = pd.read_csv(csv_tmp)
                else:
                    st.error("No table detected in PDF.")
                    file_df = None

            elif kind == "excel":
                xls = pd.ExcelFile(up)
                sheets = {nm: pd.read_excel(xls, sheet_name=nm) for nm in xls.sheet_names}
                if len(sheets) > 1:
                    file_df = _merge_excel_sheets_to_df(sheets)
                else:
                    file_df = sheets[xls.sheet_names[0]].rename_axis(None, axis=1)
                    file_df.columns = _make_unique_columns(file_df.columns)

            elif kind in ("csv", "txt"):
                file_df = read_sheet_df(up, file_kind=kind)

            else:
                st.error("Unsupported file type.")
                file_df = None

        except Exception as e:
            st.error(f"Failed to read file: {e}")
            file_df = None

        if isinstance(file_df, pd.DataFrame) and not file_df.empty:
            with st.expander("File Preview", expanded=False):
                st.dataframe(file_df.head(200), use_container_width=True, height=260)

    # ---------- DB Query ----------
    st.subheader("DB Query")
    sel_db = st.selectbox(
        "Database",
        db_options,
        index=db_options.index(default_db),
        key=get_unique_key("dbsel")
    )
    engine = get_engine(connections[sel_db]["url"])

    if st.button("Test Connection", key=get_unique_key("test_conn")):
        try:
            with engine.connect() as c:
                _ = c.execute(sqlalchemy.text("SELECT 1"))
            st.success("Connection OK")
        except Exception as e:
            st.error(f"Connection failed: {e}")

    sql = st.text_area("Enter SQL (SELECT ...)", height=110, key=get_unique_key("sql"))

    if st.button("Preview Columns", key=get_unique_key("prev_cols")):
        try:
            cols = run_query_preview(engine, sql, limit=1000)
            if cols:
                st.success("Columns: " + ", ".join(cols))
                st.session_state[get_unique_key("db_cols")] = cols
            else:
                st.warning("No columns returned.")
                st.session_state.pop(get_unique_key("db_cols"), None)
        except Exception as e:
            st.error(f"Preview failed: {e}")

    db_cols = st.session_state.get(get_unique_key("db_cols"), [])

    # ---------- Column Mapping ----------
    mapping = {}
    key_col = None
    coalesce_info = {}

    if isinstance(file_df, pd.DataFrame) and not file_df.empty and db_cols:
        st.subheader("Column Mapping")

        # DB KEY selector
        key_candidates = [c for c in db_cols if str(c).lower() in ("record_id", "id")]
        default_key = key_candidates[0] if key_candidates else db_cols[0]
        key_col = st.selectbox(
            "Select KEY column (DB)",
            db_cols,
            index=db_cols.index(default_key),
            key=get_unique_key("key_db")
        )

        # Per-column mapping (only DB columns)
        options = ["— Ignore —"] + db_cols
        cols2 = st.columns(2)
        for i, col in enumerate(file_df.columns):
            if str(col) == "source_sheet":
             continue  # skip metadata column
            with cols2[i % 2]:
                # default 'source_sheet' to Ignore
                idx = 0 if str(col) == "source_sheet" else _default_match_index(db_cols, col)
                mapped = st.selectbox(
                    f"📄 {col}",
                    options,
                    index=idx,
                    key=get_unique_key(f"map_{i}")
                )
                if mapped != "— Ignore —":
                    mapping[col] = mapped

        # Show coalesce info when multiple sources map to same target
        tgt_to_srcs = defaultdict(list)
        for src, tgt in mapping.items():
            tgt_to_srcs[tgt].append(src)
        coalesce_info = {t: s for t, s in tgt_to_srcs.items() if len(s) > 1}
        if coalesce_info:
            with st.expander("ℹ️ Multiple sources mapped to the same DB column (coalesce first non-null left→right)", expanded=True):
                for t, srcs in coalesce_info.items():
                    st.write(f"**{t}**  ⇐  {', '.join(srcs)}")

    # ---------- Controls ----------
    run_ready = (
        isinstance(file_df, pd.DataFrame) and not file_df.empty
        and bool(db_cols)
        and bool(str(sql or "").strip())
        and key_col is not None
        and any(st.session_state.get(get_unique_key(f"map_{i}")) == key_col for i in range(len(file_df.columns)))
    )
    run_btn = st.button(
        "Run Validation",
        type="primary",
        use_container_width=True,
        disabled=not run_ready,
        key=get_unique_key("run")
    )
    if not run_ready:
        st.caption("⚠️ Ensure: file uploaded • preview DB cols • enter SQL • select DB key • map the key")

    chk_dups = st.checkbox("Show Duplicates / Matches", value=True, key=get_unique_key("dups"))
    chk_nulls = st.checkbox("Show Null Issues", value=True, key=get_unique_key("nulls"))
    chk_mis   = st.checkbox("Value Mismatches (by KEY)", value=True, key=get_unique_key("mis"))

    # ---------- Run Validation ----------
    if run_btn:
        try:
            db_df = run_query_full(engine, sql)
            if db_df is None or db_df.empty:
                st.warning("DB query returned no rows.")
                return

            # Build mapped DF with coalescing for targets that have multiple sources
            tgt_to_sources_ordered = OrderedDict()
            for col in file_df.columns:  # preserve left→right order
                tgt = mapping.get(col)
                if tgt:
                    tgt_to_sources_ordered.setdefault(tgt, []).append(col)

            to_null = _to_null_factory(blank_equivs)  # reuse here
            file_mapped_cols = {}
            for tgt, srcs in tgt_to_sources_ordered.items():
                if len(srcs) == 1:
                    ser = file_df[srcs[0]]
                else:
                    tmp = file_df[srcs].copy()
                    tmp = tmp.applymap(to_null)
                    ser = tmp.bfill(axis=1).iloc[:, 0]  # first non-null left→right
                file_mapped_cols[tgt] = ser

            file_mapped = pd.DataFrame(file_mapped_cols)

            common_cols = [c for c in file_mapped.columns if c in db_df.columns]
            if key_col not in common_cols:
                st.error(f"KEY '{key_col}' must be mapped.")
                return

            file_sub = file_mapped[common_cols].copy()
            db_sub   = db_df[common_cols].copy()

            # Normalize non-key fields; coerce key to comparable string
            for c in [x for x in common_cols if x != key_col]:
                file_sub[c] = file_sub[c].map(to_null)
                db_sub[c]   = db_sub[c].map(to_null)
            file_sub[key_col] = _to_key_str(file_sub[key_col])
            db_sub[key_col]   = _to_key_str(db_sub[key_col])

            # Metrics
            m1, m2 = st.columns(2)
            m1.metric("File rows", f"{len(file_sub):,}")
            m2.metric("DB rows",   f"{len(db_sub):,}")

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

            # Duplicates / Matches
            if chk_dups:
                SENTINEL = "__NULL__"
                f_cmp = file_sub[common_cols].fillna(SENTINEL).astype(str)
                d_cmp = db_sub[common_cols].fillna(SENTINEL).astype(str)

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
            if chk_nulls:
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

            # Mismatches
            if chk_mis:
                file_dedup = file_sub.drop_duplicates(subset=[key_col])
                db_dedup   = db_sub.drop_duplicates(subset=[key_col])
                mismatch_df = compute_mismatches_by_key(file_dedup, db_dedup, key_col)
                if not mismatch_df.empty:
                    with st.expander("Value Mismatches", expanded=True):
                        st.dataframe(mismatch_df, use_container_width=True, height=300)
                    results["Value_Mismatches"] = mismatch_df

            # Report
            results["Summary"] = pd.DataFrame([
                {"metric": "File_rows", "value": len(file_sub)},
                {"metric": "DB_rows",   "value": len(db_sub)},
                {"metric": "Key",       "value": key_col},
            ])
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                for name, df in results.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df.to_excel(writer, index=False, sheet_name=str(name)[:31])
            st.download_button(
                "📥 Download Report",
                data=buf.getvalue(),
                file_name="file_to_db_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"File → DB validation failed: {e}")

    st.divider()
    if st.button("Reset Tab", key=get_unique_key("reset"), use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("file_db_"):
                del st.session_state[k]
        st.rerun()
