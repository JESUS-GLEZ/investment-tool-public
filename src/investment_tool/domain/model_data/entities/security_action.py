"""Sourced security entitlements, independent of market-data providers."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SecurityAction:
    symbol: str
    effective_at: datetime
    published_at: datetime
    cash_per_share: float
    source: str
    successor_shares: tuple[tuple[str, float], ...] = ()
    retain_original: bool = False
    # Outstanding components or unit-basis checks make the entire action unusable.
    # Known cash/share terms must not silently discharge the remaining entitlement.
    unresolved_entitlements: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.effective_at.isoformat()}"
