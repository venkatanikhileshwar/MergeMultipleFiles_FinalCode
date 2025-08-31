# utils/Multi/join_engine.py

import pandas as pd
from typing import Dict, List, Tuple, Optional
from .merge_plan_utils import MergePlan

NULL_TOKENS = {"nan", "NaN", "None", "NULL", ""}

def _normalize_str_series(s: pd.Series) -> pd.Series:
    """Convert series to string and normalize null-like values to empty string"""
    s = s.astype(str).str.strip()
    return s.where(~s.str.lower().isin({t.lower() for t in NULL_TOKENS}), "")

def harmonize_key_types(left: pd.DataFrame, right: pd.DataFrame, key_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure key columns are compatible string types with clean indexes"""
    L = left.copy().reset_index(drop=True)
    R = right.copy().reset_index(drop=True)
    
    for k in key_cols:
        if k in L.columns:
            L[k] = L[k].astype(str).str.strip()
        if k in R.columns:
            R[k] = R[k].astype(str).str.strip()
    return L, R

def _collect_conflicts(a_aligned: pd.DataFrame,
                       j_aligned: pd.DataFrame,
                       key_cols: List[str],
                       single_vs_multi: Dict[str, str]) -> pd.DataFrame:
    """
    Compare anchor vs incoming on shared single-valued columns.
    Fixed version to handle array length mismatches properly.
    """
    # Find shared columns that are single-valued (excluding keys)
    common_cols = sorted(
        c for c in set(a_aligned.columns).intersection(j_aligned.columns)
        if c not in key_cols and single_vs_multi.get(c, "single") == "single"
    )
    
    if not common_cols or a_aligned.empty or j_aligned.empty:
        return pd.DataFrame(columns=list(key_cols) + ["column", "anchor_value", "incoming_value"])

    # Ensure both dataframes have the same length
    if len(a_aligned) != len(j_aligned):
        return pd.DataFrame(columns=list(key_cols) + ["column", "anchor_value", "incoming_value"])

    out = []
    # Reset indexes to ensure positional alignment
    a_reset = a_aligned.reset_index(drop=True)
    j_reset = j_aligned.reset_index(drop=True)

    for col in common_cols:
        if col not in a_reset.columns or col not in j_reset.columns:
            continue
            
        try:
            # Normalize and align series
            av = _normalize_str_series(a_reset[col]).reset_index(drop=True)
            jv = _normalize_str_series(j_reset[col]).reset_index(drop=True)
            
            # Ensure same length
            if len(av) != len(jv):
                continue

            # Find conflicts: both non-blank AND different
            mask = (av != "") & (jv != "") & (av != jv)
            
            if not mask.any():
                continue

            # Extract key columns for conflict rows
            keys_for_conflicts = j_reset.loc[mask, key_cols].copy().reset_index(drop=True)
            
            # Create conflict data with proper alignment
            num_conflicts = mask.sum()
            conflict_data = {
                "column": [col] * num_conflicts,
                "anchor_value": av.loc[mask].values,
                "incoming_value": jv.loc[mask].values
            }
            
            # Combine keys with conflict data
            cf = keys_for_conflicts.copy()
            for key, values in conflict_data.items():
                cf[key] = values
            
            out.append(cf)
            
        except Exception as e:
            # Skip this column if there are any alignment issues
            continue

    if not out:
        return pd.DataFrame(columns=list(key_cols) + ["column", "anchor_value", "incoming_value"])

    return pd.concat(out, ignore_index=True)

def join_one(
    anchor: pd.DataFrame,
    incoming: pd.DataFrame,
    key_cols: List[str],
    how: str,
    single_vs_multi: Dict[str, str],
    source_label: str,
    plan: Optional[MergePlan] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], int]:
    """
    Join incoming dataframe to anchor dataframe.
    Returns: (working_df, conflicts_df, new_columns, key_dup_count_in_join_file)
    
    - Anchor wins for overlapping single-valued columns
    - Fills blanks from incoming
    - Multi-valued columns are left as-is for later event processing
    """
    try:
        # Harmonize key types and reset indexes
        A, J = harmonize_key_types(anchor, incoming, key_cols)
        
        # Check for key duplicates in incoming file
        kname = "__temp_key__"
        J[kname] = J[key_cols].astype(str).agg("|".join, axis=1)
        key_dup_count = int(J[kname].duplicated(keep=False).sum())

        # Get columns for joining (exclude our temp key column)
        right_cols = [c for c in J.columns if c not in key_cols and c != kname]
        
        # Perform the merge
        how_normalized = how.lower()
        merge_how = "left" if how_normalized == "left" else "inner"
        
        merged = pd.merge(
            A, 
            J[key_cols + right_cols], 
            on=key_cols, 
            how=merge_how, 
            suffixes=("", "_r")
        ).reset_index(drop=True)

        # Handle overlapping columns: Anchor wins for single-valued; fill blanks from incoming
        new_columns: List[str] = []
        
        for col in right_cols:
            if col in A.columns:
                # Overlapping column
                if single_vs_multi.get(col, "single") == "single":
                    # Single-valued: fill blanks from incoming
                    rcol = f"{col}_r"
                    if rcol in merged.columns:
                        # Fill empty/null values in anchor with values from incoming
                        anchor_empty = (
                            merged[col].isna() | 
                            (merged[col].astype(str).str.strip() == "") |
                            (merged[col].astype(str).str.lower().isin(NULL_TOKENS))
                        )
                        merged.loc[anchor_empty, col] = merged.loc[anchor_empty, rcol]
                        merged.drop(columns=[rcol], inplace=True)
                else:
                    # Multi-valued: keep anchor, fill blanks if needed
                    rcol = f"{col}_r"
                    if rcol in merged.columns:
                        anchor_empty = (
                            merged[col].isna() | 
                            (merged[col].astype(str).str.strip() == "")
                        )
                        merged.loc[anchor_empty, col] = merged.loc[anchor_empty, rcol]
                        merged.drop(columns=[rcol], inplace=True)
            else:
                # Brand new column from incoming
                new_columns.append(col)

        # Collect conflicts (only for single-valued columns)
        try:
            # Create aligned slices for conflict detection
            A_aligned = merged[[c for c in A.columns if c in merged.columns]].copy()
            
            # Re-merge incoming data to get aligned version for conflict detection
            J_for_conflicts = pd.merge(
                merged[key_cols], 
                J[key_cols + right_cols], 
                on=key_cols, 
                how="left"
            ).reset_index(drop=True)
            
            conflicts_df = _collect_conflicts(A_aligned, J_for_conflicts, key_cols, single_vs_multi)
            
        except Exception as e:
            # If conflict detection fails, return empty conflicts
            conflicts_df = pd.DataFrame(columns=key_cols + ["column", "anchor_value", "incoming_value"])

        # Update merge plan
        if plan is not None:
            plan.add_step(f"⚡ {how.upper()} with {source_label} — rows now: {len(merged):,} — new cols: {len(new_columns)} — key dups in join: {key_dup_count}")
            if len(merged) == 0:
                plan.warn("Join produced 0 rows. Consider switching to LEFT or a different key.")

        return merged, conflicts_df, new_columns, key_dup_count
        
    except Exception as e:
        # Return original anchor and empty results if join fails completely
        empty_conflicts = pd.DataFrame(columns=key_cols + ["column", "anchor_value", "incoming_value"])
        if plan is not None:
            plan.warn(f"Join with {source_label} failed: {str(e)}")
        return anchor.copy().reset_index(drop=True), empty_conflicts, [], 0