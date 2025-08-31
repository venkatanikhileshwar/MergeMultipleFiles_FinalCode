
# ================================================
# filepath: utils/Multi/export_utils.py
# ================================================
from __future__ import annotations
import io
import pandas as pd
from typing import Optional

try:
    # your project writer
    from report_utils import export_excel as _export_excel
except Exception:
    _export_excel = None

__all__ = ["export_join_outputs"]


def export_join_outputs(
    patient_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame],
    conflicts_df: Optional[pd.DataFrame],
    plan_text: str,
    path: str,
) -> str:
    """Bundle outputs into a single XLSX if writer available, else CSV fallbacks.
    Returns the suggested filename/path.
    """
    if _export_excel is not None:
        sheets = {"Patient_1to1": patient_df}
        if events_df is not None and not events_df.empty:
            sheets["Clinical_Events_1toMany"] = events_df
        if conflicts_df is not None and not conflicts_df.empty:
            sheets["Conflicts"] = conflicts_df
        # add plan as a tiny sheet
        plan_df = pd.DataFrame({"Merge Plan": plan_text.splitlines()})
        sheets["Merge_Plan"] = plan_df
        _export_excel(sheets, path)
        return path
    # Fallback: write individual CSVs alongside a .txt
    patient_path = path.replace(".xlsx", "_patient.csv")
    patient_df.to_csv(patient_path, index=False)
    if events_df is not None and not events_df.empty:
        events_df.to_csv(path.replace(".xlsx", "_events.csv"), index=False)
    if conflicts_df is not None and not conflicts_df.empty:
        conflicts_df.to_csv(path.replace(".xlsx", "_conflicts.csv"), index=False)
    with open(path.replace(".xlsx", "_plan.txt"), "w", encoding="utf-8") as f:
        f.write(plan_text)
    return patient_path
