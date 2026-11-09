"""Load reviewable stock and crypto universes from ``config/*.json``."""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from investment_tool.data.universes.models import (
    AssetUniverse,
    DatedMembership,
    MembershipSnapshot,
)

from investment_tool.domain.model_data.entities.cash_settlement import CashSettlement
from investment_tool.domain.model_data.entities.security_action import SecurityAction

DEFAULT_ASSET_CONFIG_FILE = "example_asset_universe.json"
ASSET_CONFIG_DIR = Path(__file__).resolve().parents[4] / "config"
_STOCK = re.compile(r"^[A-Z][A-Z0-9.-]*$")
_CRYPTO = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")


def _instant(value: Any) -> datetime:
    """Require an explicit timezone for membership and publication boundaries."""
    if not isinstance(value, str):
        raise ValueError("Membership timestamps must be timezone-aware ISO strings")
    try:
        instant = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Invalid membership timestamp") from exc
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("Membership timestamps must include a timezone")
    return instant.astimezone(timezone.utc)


def _load_membership(data: Any, stocks: list[str]) -> DatedMembership | None:
    if data is None:
        return None
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Unsupported dated membership format")
    minimum = data.get("minimum_return_periods")
    limit = data.get("stock_limit")
    if type(minimum) is not int or minimum < 2:
        raise ValueError("minimum_return_periods must be an integer >= 2")
    if type(limit) is not int or limit < 1:
        raise ValueError("stock_limit must be a positive integer")
    fixed = data.get("always_considered_stocks", [])
    if not isinstance(fixed, list) or any(not isinstance(s, str) for s in fixed):
        raise ValueError("Invalid always-considered stock symbols")
    if len(fixed) != len(set(fixed)) or not set(fixed) <= set(stocks):
        raise ValueError("Always-considered symbols must be unique configured stocks")
    items = data.get("snapshots")
    if not isinstance(items, list) or not items:
        raise ValueError("Dated membership requires nonempty snapshots")
    snapshots: list[MembershipSnapshot] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid membership snapshot")
        start = _instant(item.get("effective_from"))
        end = _instant(item.get("expires_at"))
        published = _instant(item.get("published_at"))
        symbols = item.get("symbols")
        source = item.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Each membership snapshot requires a source")
        if published > start or start >= end:
            raise ValueError(
                "Snapshot must be published before taking effect and expire later"
            )
        if snapshots and start < snapshots[-1].expires_at:
            raise ValueError("Membership snapshots must be ordered without overlaps")
        if (
            not isinstance(symbols, list)
            or not symbols
            or any(not isinstance(s, str) for s in symbols)
        ):
            raise ValueError("Snapshot symbols must be a nonempty stock list")
        if len(symbols) != len(set(symbols)) or len(symbols) > limit:
            raise ValueError("Snapshot has duplicate symbols or exceeds stock_limit")
        if not set(symbols) <= set(stocks) or set(symbols) & set(fixed):
            raise ValueError(
                "Snapshot equities must be configured stocks separate from fixed candidates"
            )
        snapshots.append(
            MembershipSnapshot(start, end, published, frozenset(symbols), source)
        )
    union = set(fixed).union(*(s.symbols for s in snapshots))
    if union != set(stocks):
        raise ValueError(
            "Configured stocks must equal the union of all dated candidates"
        )
    return DatedMembership(minimum, limit, frozenset(fixed), tuple(snapshots))


def _history_boundaries(data: Any, symbols: list[str]) -> dict[str, date]:
    if not isinstance(data, dict) or not set(data) <= set(symbols):
        raise ValueError("Price history boundaries must reference configured assets")
    boundaries: dict[str, date] = {}
    for symbol, value in data.items():
        if not isinstance(value, str):
            raise ValueError("Price history boundaries must be ISO calendar dates")
        try:
            boundary = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Invalid price history boundary") from exc
        if value != boundary.isoformat():
            raise ValueError("Price history boundaries must use YYYY-MM-DD")
        boundaries[symbol] = boundary
    return boundaries


def _unresolved_entitlements(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(reason, str) or not reason.strip() for reason in value
    ):
        raise ValueError("unresolved_entitlements must be a list of nonempty strings")
    reasons = tuple(reason.strip() for reason in value)
    if len(set(reasons)) != len(reasons):
        raise ValueError("unresolved_entitlements must not contain duplicates")
    return reasons


def _load_cash_settlements(data: Any, symbols: list[str]) -> tuple[CashSettlement, ...]:
    if not isinstance(data, list):
        raise ValueError("cash_settlements must be a list")
    events: list[CashSettlement] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Invalid cash settlement")
        symbol = item.get("symbol")
        source = item.get("source")
        amount = item.get("cash_per_share")
        if not isinstance(symbol, str) or symbol not in symbols or symbol in seen:
            raise ValueError("Cash settlement must reference a unique configured stock")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Cash settlement requires a reviewed source")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError("Cash settlement must have a finite nonnegative amount")
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Cash settlement must have a finite nonnegative amount")
        effective = _instant(item.get("effective_at"))
        published = _instant(item.get("published_at"))
        events.append(
            CashSettlement(
                symbol,
                effective,
                published,
                float(amount),
                source,
                unresolved_entitlements=_unresolved_entitlements(
                    item.get("unresolved_entitlements", [])
                ),
            )
        )
        seen.add(symbol)
    return tuple(sorted(events, key=lambda e: e.effective_at))


def _load_security_actions(data: Any, symbols: list[str]) -> tuple[SecurityAction, ...]:
    if not isinstance(data, list):
        raise ValueError("security_actions must be a list")
    events: list[SecurityAction] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Invalid security action")
        symbol, source = item.get("symbol"), item.get("source")
        amount = item.get("cash_per_share", 0)
        shares = item.get("successor_shares", {})
        retain = item.get("retain_original", False)
        reasons = _unresolved_entitlements(item.get("unresolved_entitlements", []))
        if (
            not isinstance(symbol, str)
            or symbol not in symbols
            or not isinstance(source, str)
            or not source.strip()
        ):
            raise ValueError(
                "Security action requires a configured security and source"
            )
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
            or amount < 0
        ):
            raise ValueError("Invalid security-action cash amount")
        if (
            type(retain) is not bool
            or not isinstance(shares, dict)
            or (not shares and not reasons)
        ):
            raise ValueError(
                "Security action requires successor shares or unresolved entitlements "
                "and a boolean retain_original"
            )
        ratios: list[tuple[str, float]] = []
        for successor, ratio in shares.items():
            if successor not in symbols or successor == symbol:
                raise ValueError(
                    "Successor must reference another configured quote security"
                )
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio)
                or ratio <= 0
            ):
                raise ValueError("Successor share ratios must be finite and positive")
            ratios.append((successor, float(ratio)))
        event = SecurityAction(
            symbol,
            _instant(item.get("effective_at")),
            _instant(item.get("published_at")),
            float(amount),
            source,
            tuple(sorted(ratios)),
            retain,
            reasons,
        )
        if event.key in seen:
            raise ValueError("Duplicate security action")
        seen.add(event.key)
        events.append(event)
    return tuple(
        sorted(
            events,
            key=lambda event: (max(event.effective_at, event.published_at), event.key),
        )
    )


def load_asset_universe(filename: str = DEFAULT_ASSET_CONFIG_FILE) -> AssetUniverse:
    """Load an active or archived manifest by basename for saved-run compatibility."""
    if (
        not filename
        or Path(filename).name != filename
        or not filename.endswith(".json")
    ):
        raise ValueError("asset_config_file must be a JSON filename in config/")
    path = ASSET_CONFIG_DIR / filename
    if path.resolve().parent != ASSET_CONFIG_DIR.resolve():
        raise ValueError("asset_config_file must stay inside config/")
    if not path.is_file():
        path = ASSET_CONFIG_DIR / "archive" / filename
        if path.resolve().parent != ASSET_CONFIG_DIR.resolve() / "archive":
            raise ValueError("asset_config_file must stay inside config/archive/")
    if not path.is_file():
        raise ValueError(f"Asset configuration '{filename}' does not exist")
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read asset configuration '{filename}'") from exc
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        raise ValueError(f"Invalid asset configuration '{filename}': missing name")
    groups = data.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(
            f"Invalid asset configuration '{filename}': missing stock groups"
        )
    stocks: list[str] = []
    for group in groups.values():
        if not isinstance(group, list):
            raise ValueError(
                f"Invalid asset configuration '{filename}': stock group must be a list"
            )
        stocks.extend(group)
    crypto = data.get("crypto_symbols")
    if not isinstance(crypto, list):
        raise ValueError(
            f"Invalid asset configuration '{filename}': crypto_symbols must be a list"
        )
    if not stocks or any(
        not isinstance(s, str) or not _STOCK.fullmatch(s) for s in stocks
    ):
        raise ValueError(f"Invalid stock symbols in '{filename}'")
    if any(not isinstance(s, str) or not _CRYPTO.fullmatch(s) for s in crypto):
        raise ValueError(
            f"Invalid crypto symbols in '{filename}'; use BASE/QUOTE pairs"
        )
    if len(set(stocks)) != len(stocks) or len(set(crypto)) != len(crypto):
        raise ValueError(f"Duplicate symbols in '{filename}'")
    if "symbol_count" in data and data["symbol_count"] != len(stocks):
        raise ValueError(f"Stock symbol_count does not match groups in '{filename}'")
    valuation = data.get("valuation_symbols", [])
    if not isinstance(valuation, list) or any(
        not isinstance(s, str) or not _STOCK.fullmatch(s) for s in valuation
    ):
        raise ValueError("Invalid valuation symbols")
    if len(set(valuation)) != len(valuation) or set(valuation) & set(stocks):
        raise ValueError(
            "Valuation symbols must be unique and separate from candidates"
        )
    quote_stocks = stocks + valuation
    history_starts = _history_boundaries(
        data.get("price_history_starts", {}), quote_stocks + crypto
    )
    history_ends = _history_boundaries(
        data.get("price_history_ends", {}), quote_stocks + crypto
    )
    if any(
        history_starts[s] >= history_ends[s]
        for s in history_starts.keys() & history_ends.keys()
    ):
        raise ValueError("Price history start must precede its exclusive end")
    history_sources = data.get("price_history_sources", {})
    if (
        not isinstance(history_sources, dict)
        or not set(history_sources) <= (history_starts.keys() | history_ends.keys())
        or any(
            not isinstance(value, str) or not value.strip()
            for value in history_sources.values()
        )
    ):
        raise ValueError(
            "Price history sources must reference bounded assets with nonempty sources"
        )
    membership = _load_membership(data.get("membership"), stocks)
    minimum = data.get("minimum_return_periods")
    if minimum is not None and (type(minimum) is not int or minimum < 2):
        raise ValueError("minimum_return_periods must be an integer >= 2")
    if membership is not None:
        if minimum is not None and minimum != membership.minimum_return_periods:
            raise ValueError("Conflicting minimum return history in manifest")
        minimum = membership.minimum_return_periods
    identity_value = data.get("symbol_identity_date")
    identity_date = None
    if identity_value is not None:
        if not isinstance(identity_value, str):
            raise ValueError("symbol_identity_date must use YYYY-MM-DD")
        try:
            identity_date = date.fromisoformat(identity_value)
        except ValueError as exc:
            raise ValueError("Invalid symbol_identity_date") from exc
        if identity_value != identity_date.isoformat():
            raise ValueError("symbol_identity_date must use YYYY-MM-DD")
    cash_events = _load_cash_settlements(data.get("cash_settlements", []), stocks)
    share_events = _load_security_actions(
        data.get("security_actions", []), quote_stocks
    )
    if {event.key for event in cash_events} & {event.key for event in share_events}:
        raise ValueError("Duplicate corporate action across cash and share records")
    return AssetUniverse(
        filename,
        data["name"],
        stocks,
        crypto,
        membership,
        history_starts,
        minimum,
        identity_date,
        history_ends,
        cash_events,
        share_events,
        tuple(valuation),
        history_sources,
    )


def list_asset_universes() -> list[AssetUniverse]:
    """Return the validated manifests available to new runs."""
    return [
        load_asset_universe(p.name) for p in sorted(ASSET_CONFIG_DIR.glob("*.json"))
    ]
