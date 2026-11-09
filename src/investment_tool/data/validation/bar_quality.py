"""Quarantine unusable observations without removing a security's history."""

from typing import Any

import numpy as np
import pandas as pd

from .validators import BaseValidator
from .validation_report import IssueSeverity, ValidationIssue, ValidationReport


class BarQualityValidator(BaseValidator):
    """Invalid rows become missing observations, never synthetic prices.

    Large but internally consistent returns are left intact. Valid earlier and
    later rows remain available; eligibility is assessed at each decision.
    """

    name = "BarQualityValidator"

    def validate(
        self, data: dict[str, Any], report: ValidationReport
    ) -> dict[str, Any]:
        for key in ("stock_dict", "crypto_dict"):
            for symbol, frame in data.get(key, {}).items():
                columns = ["open", "high", "low", "close", "volume"]
                if not set(columns) <= set(frame.columns):
                    continue  # RequiredColumnsValidator reports structural issues.
                values = (
                    frame[columns].apply(pd.to_numeric, errors="coerce").astype(float)
                )
                prices = values[columns[:4]]
                invalid_volume = ~np.isfinite(values["volume"]) | (
                    values["volume"] <= 0
                )
                invalid_open = ~np.isfinite(values["open"]) | (values["open"] <= 0)
                invalid = ~np.isfinite(values).all(axis=1)
                invalid |= (prices <= 0).any(axis=1) | invalid_volume
                invalid |= values["low"] > prices[["open", "close", "high"]].min(axis=1)
                invalid |= values["high"] < prices[["open", "close", "low"]].max(axis=1)
                cleaned = frame.copy()
                cleaned[columns] = values
                if not values.dtypes.equals(frame[columns].dtypes):
                    report.record_modification(
                        symbol, "Normalized numeric OHLCV fields"
                    )
                data[key][symbol] = cleaned
                if not invalid.any():
                    continue
                # Future high/low/close quality must never decide whether an
                # otherwise valid opening quote could be bought that morning.
                # Quarantine the completed-bar features; the readiness audit
                # blocks unresolved valuation instead of excluding by hindsight.
                cleaned.loc[invalid, ["high", "low", "close"]] = np.nan
                cleaned.loc[invalid_open | invalid_volume, "open"] = np.nan
                cleaned.loc[invalid_volume, "volume"] = np.nan
                report.add_issue(
                    ValidationIssue(
                        IssueSeverity.WARNING,
                        self.name,
                        "Unusable OHLCV observations quarantined; no prices filled",
                        symbol=symbol,
                        action="quarantine_observations",
                        details={
                            "dates": [str(d) for d in frame.index[invalid]],
                            "count": int(invalid.sum()),
                        },
                    )
                )
                report.record_modification(symbol, "Invalid OHLCV rows set to missing")
        return data
