"""Shared artifact policy; never controls search, selection or checkpoints."""

from enum import Enum


class OutputMode(str, Enum):
    STANDARD = "standard"
    DIAGNOSTIC = "diagnostic"

    @property
    def save_chart_exports(self) -> bool:
        return self == OutputMode.DIAGNOSTIC
