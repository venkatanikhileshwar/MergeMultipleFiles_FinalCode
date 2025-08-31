
# ================================================
# filepath: utils/Multi/merge_plan_utils.py
# ================================================
from __future__ import annotations
from typing import List

__all__ = ["MergePlan"]

class MergePlan:
    def __init__(self) -> None:
        self._lines: List[str] = []
    def start(self, mode: str, key_cols: list[str] | None = None, how: str | None = None) -> None:
        if mode.lower() == "append":
            self._lines.append(f"Mode: APPEND (stack rows)")
        else:
            key_txt = ", ".join(key_cols or [])
            self._lines.append(f"Mode: JOIN on [{key_txt}] • Type: {how}")
    def add_step(self, text: str) -> None:
        self._lines.append(text)
    def warn(self, text: str) -> None:
        self._lines.append(f"⚠ {text}")
    def to_text(self) -> str:
        return "\n".join(self._lines)