# tabs/db_to_db.py
import streamlit as st
import pandas as pd
import json
import io
import sqlalchemy
from collections import defaultdict

from utils.sql_utils import get_engine, run_query_preview, run_query_full
from utils.validate_utils import compute_mismatches_by_key


# ------------ helpers ------------
def get_unique_key(base_key: str) -> str:
    return f"dbdb_{base_key}"


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


def _default_match_index(target_cols, src_col):
    if not target_cols:
        return 0
    for i, c in enumerate(target_cols):
        if str(c).lower() == str(src_col).lower():
            return i + 1  # options[0] is "— Ignore —"
    return 0


# ------------ main tab ------------
def render():
    cfg = _load_cfg()
    db_options = cfg["defaults"]["db"]["dropdown"]
    default_db = cfg["defaults"]["db"]["default"]
    connections = cfg["connections"]
    blank_equivs = set(cfg.get("normalization", {}).get("blank_equivalents", []))
    to_null = _to_null_factory(blank_equivs)

    st.header("DB ↔ DB")

    # --- DB A (source) ---
    st.subheader("DB A")
    sel_a = st.selectbox("Database A", db_options, index=db_options.index(default_db), key=get_unique_key("dbA"))
    eng_a = get_engine(connections[sel_a]["url"])

    if st.button("Test Connection A", key=get_unique_key("testA")):
        try:
            with eng_a.connect() as c:
                _ = c.execute(sqlalchemy.text("SELECT 1"))
            st.success("A: Connection OK")
        except Exception as e:
            st.error(f"A: Connection failed: {e}")

    sql_a = st.text_area("SQL for DB A (SELECT ...)", height=110, key=get_unique_key("sqlA"))

    if st.button("Preview Columns A", key=get_unique_key("prevA")):
        try:
            cols = run_query_preview(eng_a, sql_a, limit=1000)
            if cols:
                st.success("A Columns: " + ", ".join(cols))
                st.session_state[get_unique_key("colsA")] = cols
            else:
                st.warning("A returned no columns.")
                st.session_state.pop(get_unique_key("colsA"), None)
        except Exception as e:
            st.error(f"Preview A failed: {e}")

    cols_a = st.session_state.get(get_unique_key("colsA"), [])

    st.markdown("---")

    # --- DB B (target) ---
    st.subheader("DB B")
    sel_b = st.selectbox("Database B", db_options, index=db_options.index(default_db), key=get_unique_key("dbB"))
    eng_b = get_engine(connections[sel_b]["url"])

    if st.button("Test Connection B", key=get_unique_key("testB")):
        try:
            with eng_b.connect() as c:
                _ = c.execute(sqlalchemy.text("SELECT 1"))
            st.success("B: Connection OK")
        except Exception as e:
            st.error(f"B: Connection failed: {e}")

    sql_b = st.text_area("SQL for DB B (SELECT ...)", height=110, key=get_unique_key("sqlB"))

    if st.button("Preview Columns B", key=get_unique_key("prevB")):
        try:
            cols = run_query_preview(eng_b, sql_b, limit=1000)
            if cols:
                st.success("B Columns: " + ", ".join(cols))
                st.session_state[get_unique_key("colsB")] = cols
            else:
                st.warning("B returned no columns.")
                st.session_state.pop(get_unique_key("colsB"), None)
        except Exception as e:
            st.error(f"Preview B failed: {e}")

    cols_b = st.session_state.get(get_unique_key("colsB"), [])

    # --- Mapping (A → B), B key picker ---
    mapping = {}
    key_col_b = None

    if cols_a and cols_b:
        st.subheader("Column Mapping  (DB A → DB B)")
        # choose KEY from B (target)
        key_candidates = [c for c in cols_b if str(c).lower() in ("record_id", "id")]
        default_key = key_candidates[0] if key_candidates else cols_b[0]
        key_col_b = st.selectbox(
            "Select KEY column (in DB B)",
            cols_b,
            index=cols_b.index(default_key),
            key=get_unique_key("keyB")
        )

        options = ["— Ignore —"] + cols_b
        cols2 = st.columns(2)
        for i, col in enumerate(cols_a):
            with cols2[i % 2]:
                idx = _default_match_index(cols_b, col)
                mapped = st.selectbox(
                    f"A: {col}",
                    options,
                    index=idx,
                    key=get_unique_key(f"map_{i}")
                )
                if mapped != "— Ignore —":
                    mapping[col] = mapped

        # Informative expander if multiple A columns target same B column (we allow & coalesce)
        from collections import defaultdict as dfd
        tgt_to_srcs = dfd(list)
        for src, tgt in mapping.items():
            tgt_to_srcs[tgt].append(src)
        multi = {t: s for t, s in tgt_to_srcs.items() if len(s) > 1}
        if multi:
            with st.expander("ℹ️ Multiple A columns mapped to the same B column (will coalesce left→right)", expanded=False):
                for t, srcs in multi.items():
                    st.write(f"**{t}**  ⇐  {', '.join(srcs)}")

    # --- Controls (enable only when ready) ---
    run_ready = (
        bool(cols_a) and bool(cols_b)
        and bool(str(sql_a or "").strip()) and bool(str(sql_b or "").strip())
        and key_col_b is not None
        and any(st.session_state.get(get_unique_key(f"map_{i}")) == key_col_b for i in range(len(cols_a)))
    )

    run_btn = st.button(
        "Run DB ↔ DB Validation",
        type="primary",
        use_container_width=True,
        disabled=not run_ready,
        key=get_unique_key("run")
    )
    if not run_ready:
        st.caption("⚠️ Do this first: Preview A • Preview B • enter both SQLs • pick B key • map at least one A column to that key")

    chk_dups = st.checkbox("Show Duplicates / Matches (full-row)", value=True, key=get_unique_key("dups"))
    chk_nulls = st.checkbox("Show Null Issues (A & B)", value=True, key=get_unique_key("nulls"))
    chk_mis   = st.checkbox("Value Mismatches (by KEY)", value=True, key=get_unique_key("mis"))

    # --- Run ---
    if run_btn:
        try:
            df_a = run_query_full(eng_a, sql_a)
            df_b = run_query_full(eng_b, sql_b)

            if df_a is None or df_a.empty:
                st.warning("DB A returned no rows.")
                return
            if df_b is None or df_b.empty:
                st.warning("DB B returned no rows.")
                return

            # Build A_mapped with B's column names (coalesce when multiple A sources → one B target)
            from collections import OrderedDict
            tgt_to_sources_ordered = OrderedDict()
            for col in cols_a:  # preserve left→right
                tgt = mapping.get(col)
                if tgt:
                    tgt_to_sources_ordered.setdefault(tgt, []).append(col)

            file_mapped_cols = {}
            for tgt, srcs in tgt_to_sources_ordered.items():
                if len(srcs) == 1:
                    ser = df_a[srcs[0]] if srcs[0] in df_a.columns else pd.Series(pd.NA, index=df_a.index)
                else:
                    tmp_srcs = [c for c in srcs if c in df_a.columns]
                    if not tmp_srcs:
                        ser = pd.Series(pd.NA, index=df_a.index)
                    else:
                        tmp = df_a[tmp_srcs].copy()
                        tmp = tmp.applymap(to_null)
                        ser = tmp.bfill(axis=1).iloc[:, 0]
                file_mapped_cols[tgt] = ser

            a_mapped = pd.DataFrame(file_mapped_cols)  # columns named like B
            # Work on common columns only (intersection of a_mapped and df_b)
            common_cols = [c for c in a_mapped.columns if c in df_b.columns]
            if key_col_b not in common_cols:
                st.error(f"KEY '{key_col_b}' must be mapped from A.")
                return

            A = a_mapped[common_cols].copy()
            B = df_b[common_cols].copy()

            # Normalize non-key cols; coerce key to comparable string
            for c in [x for x in common_cols if x != key_col_b]:
                A[c] = A[c].map(to_null)
                B[c] = B[c].map(to_null)
            A[key_col_b] = _to_key_str(A[key_col_b])
            B[key_col_b] = _to_key_str(B[key_col_b])

            m1, m2 = st.columns(2)
            m1.metric("DB A rows", f"{len(A):,}")
            m2.metric("DB B rows", f"{len(B):,}")

            results = {}

            # Missing keys
            keys_a = set(A[key_col_b].dropna())
            keys_b = set(B[key_col_b].dropna())
            missing_in_b = A[A[key_col_b].isin(keys_a - keys_b)]
            missing_in_a = B[B[key_col_b].isin(keys_b - keys_a)]
            if not missing_in_b.empty:
                with st.expander(f"Missing in B ({len(missing_in_b)})"):
                    st.dataframe(missing_in_b, use_container_width=True, height=240)
                results["Missing_in_B"] = missing_in_b
            if not missing_in_a.empty:
                with st.expander(f"Missing in A ({len(missing_in_a)})"):
                    st.dataframe(missing_in_a, use_container_width=True, height=240)
                results["Missing_in_A"] = missing_in_a

            # Duplicates / Matches
            if chk_dups:
                SENTINEL = "__NULL__"
                a_cmp = A[common_cols].fillna(SENTINEL).astype(str)
                b_cmp = B[common_cols].fillna(SENTINEL).astype(str)

                exact = pd.merge(
                    a_cmp.drop_duplicates(),
                    b_cmp.drop_duplicates(),
                    on=common_cols, how="inner"
                )
                if not exact.empty:
                    with st.expander(f"Exact Matches ({len(exact)})", expanded=False):
                        st.dataframe(exact, use_container_width=True, height=240)
                    results["Exact_Matches"] = exact

                both = pd.concat([a_cmp.assign(_side="A"),
                                  b_cmp.assign(_side="B")], ignore_index=True)
                counts = (both.groupby(common_cols)["_side"]
                          .value_counts().unstack(fill_value=0).reset_index())
                if "A" not in counts.columns: counts["A"] = 0
                if "B" not in counts.columns: counts["B"] = 0
                counts["extras"] = counts["A"].sub(1).clip(lower=0) + counts["B"].sub(1).clip(lower=0)
                extras = counts[counts["extras"] > 0]
                if not extras.empty:
                    with st.expander(f"Extra Duplicates ({len(extras)})", expanded=True):
                        st.dataframe(extras, use_container_width=True, height=280)
                    results["Duplicates"] = extras

            # Null Issues
            if chk_nulls:
                chk_cols = [c for c in common_cols if c != key_col_b]
                a_nulls = A[A[chk_cols].isna().any(axis=1)]
                b_nulls = B[B[chk_cols].isna().any(axis=1)]
                if not a_nulls.empty:
                    with st.expander(f"DB A Null Issues ({len(a_nulls)})"):
                        st.dataframe(a_nulls, use_container_width=True, height=220)
                    results["A_Null_Issues"] = a_nulls
                if not b_nulls.empty:
                    with st.expander(f"DB B Null Issues ({len(b_nulls)})"):
                        st.dataframe(b_nulls, use_container_width=True, height=220)
                    results["B_Null_Issues"] = b_nulls

            # Value mismatches by KEY
            if chk_mis:
                a_dedup = A.drop_duplicates(subset=[key_col_b])
                b_dedup = B.drop_duplicates(subset=[key_col_b])
                mismatch_df = compute_mismatches_by_key(a_dedup, b_dedup, key_col_b)
                if not mismatch_df.empty:
                    with st.expander("Value Mismatches", expanded=True):
                        st.dataframe(mismatch_df, use_container_width=True, height=300)
                    results["Value_Mismatches"] = mismatch_df

            # Excel report
            results["Summary"] = pd.DataFrame([
                {"metric": "A_rows", "value": len(A)},
                {"metric": "B_rows", "value": len(B)},
                {"metric": "Key (B)", "value": key_col_b},
            ])
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                for name, df in results.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df.to_excel(writer, index=False, sheet_name=str(name)[:31])
            st.download_button(
                "📥 Download Report",
                data=buf.getvalue(),
                file_name="db_to_db_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"DB ↔ DB validation failed: {e}")

    st.divider()
    if st.button("Reset Tab", key=get_unique_key("reset"), use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("dbdb_"):
                del st.session_state[k]
        st.rerun()
