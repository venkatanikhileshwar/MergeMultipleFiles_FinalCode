# ================================================
# filepath: utils/Multi/key_suggest_utils.py
# ================================================
from __future__ import annotations
import re
import pandas as pd
from typing import Dict, List

__all__ = [
    "find_candidate_keys",
    "score_keys",
    "recommend_default_key",
]

_ID_HINT = re.compile(r"(^|_|\b)(id|patient|mrn|encounter|visit|record|roll)(_|\b)")

def _non_blank_count(s: pd.Series) -> int:
    s = s.astype(str).map(lambda x: x.strip())
    s = s.replace(to_replace=["nan", "NaN", "None", "NULL"], value="")
    return s.astype(bool).sum()

def find_candidate_keys(dfs: Dict[str, pd.DataFrame]) -> List[List[str]]:
    cols = set()
    for df in dfs.values():
        cols |= set(map(str, df.columns))
    singles = [[c] for c in sorted(cols)]
    singles.sort(key=lambda k: (0 if _ID_HINT.search(k[0]) else 1, k[0]))
    return singles

def score_keys(dfs: Dict[str, pd.DataFrame], keys: List[List[str]]) -> List[Dict]:
    max_rows = max((len(df) for df in dfs.values()), default=0)
    scored = []
    for key_cols in keys:
        coverage_parts = []
        dup_per_file = []
        present_in = []
        for name, df in dfs.items():
            has_all = all(k in df.columns for k in key_cols)
            present_in.append(name if has_all else None)
            if not has_all or max_rows == 0:
                coverage_parts.append(0.0)
                dup_per_file.append(0)
                continue
            subset = df[key_cols].astype(str).fillna("").copy()
            subset["__k__"] = subset.apply(lambda r: "|".join(r.values.tolist()), axis=1)
            nb = _non_blank_count(subset["__k__"])
            coverage_parts.append(nb / max_rows)
            dup_per_file.append(int(subset["__k__"].duplicated(keep=False).sum()))
        coverage = 100.0 * (sum(coverage_parts) / (len(coverage_parts) or 1))
        is_id = any(_ID_HINT.search(k) for k in key_cols)
        scored.append({
            "key": key_cols,
            "coverage_pct": round(coverage, 1),
            "dup_counts": dup_per_file,
            "present_in": [p for p in present_in if p],
            "identifier": is_id,
            "not_recommended": (not is_id),
        })
    scored.sort(key=lambda x: (
        0 if x["identifier"] else 1,
        -x["coverage_pct"],
        sum(x["dup_counts"]),
        len(x["key"]),
    ))
    return scored

def recommend_default_key(scored: List[Dict]) -> List[str]:
    return (scored[0]["key"] if scored else [])
