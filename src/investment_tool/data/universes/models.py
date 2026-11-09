"""Universe definitions and dated candidate membership."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from investment_tool.domain.model_data.entities.cash_settlement import CashSettlement
from investment_tool.domain.model_data.entities.security_action import SecurityAction


@dataclass(frozen=True)
class MembershipSnapshot:
    effective_from: datetime
    expires_at: datetime
    published_at: datetime
    symbols: frozenset[str]
    source: str


@dataclass(frozen=True)
class DatedMembership:
    """Reviewed quarterly candidate lists, with bounded historical coverage."""

    minimum_return_periods: int
    stock_limit: int
    always_considered: frozenset[str]
    snapshots: tuple[MembershipSnapshot, ...]

    def symbols_at(self, decision_time: datetime) -> frozenset[str]:
        if decision_time.tzinfo is None or decision_time.utcoffset() is None:
            raise ValueError("Dated universes require a timezone-aware decision")
        decision = decision_time.astimezone(timezone.utc)
        for snapshot in self.snapshots:
            if snapshot.effective_from <= decision < snapshot.expires_at:
                if snapshot.published_at > decision:
                    raise ValueError("Universe snapshot was not published at decision")
                return snapshot.symbols | self.always_considered
        raise ValueError(
            f"No dated universe snapshot covers {decision.isoformat()}; "
            "provide the missing historical snapshot rather than reuse a later list"
        )


@dataclass(frozen=True)
class AssetUniverse:
    file: str
    name: str
    stocks: list[str]
    crypto: list[str]
    membership: DatedMembership | None = None
    price_history_starts: dict[str, date] = field(default_factory=dict)
    minimum_return_periods: int | None = None
    symbol_identity_date: date | None = None
    price_history_ends: dict[str, date] = field(default_factory=dict)

    cash_settlements: tuple[CashSettlement, ...] = ()
    security_actions: tuple[SecurityAction, ...] = ()
    valuation_symbols: tuple[str, ...] = ()
    price_history_sources: dict[str, str] = field(default_factory=dict)

    @property
    def quote_stocks(self) -> list[str]:
        """Prices required for candidates and received holdings, in stable order."""
        return [*self.stocks, *self.valuation_symbols]

    def summary(self) -> dict[str, str | int]:
        summary: dict[str, str | int] = {
            "file": self.file,
            "name": self.name,
            "stock_count": len(self.stocks),
            "crypto_count": len(self.crypto),
        }
        if self.membership is not None:
            summary["membership_mode"] = "dated"
            summary["stock_limit"] = self.membership.stock_limit
        return summary
