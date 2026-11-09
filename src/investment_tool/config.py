"""
Configuration management for the Investment Tool.

This module loads configuration from multiple sources with the following
precedence (highest to lowest):
1. Environment variables (ALPACA_API_KEY, ALPACA_API_SECRET, ALPHA_VANTAGE_API_KEY)
2. Local config file (config.local.ini) if present
3. Default config file (config.ini bundled with package)

Usage::

    from investment_tool.config import load_config, AppConfig
    cfg = load_config()           # returns a frozen AppConfig dataclass
    cfg.alpaca.api_key            # typed, IDE-completable

For convenience the module exposes a lazy singleton via :func:`get_config`
that is built on first access.  All downstream code should accept an
explicit ``cfg`` parameter and use ``get_config()`` only as a last-resort
default (this keeps the code testable and multi-tenant-ready).
"""

from __future__ import annotations

import configparser
import importlib.resources
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

from investment_tool.data.universes.loader import DEFAULT_ASSET_CONFIG_FILE
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlpacaConfig:
    """Alpaca API credentials and endpoint."""

    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
    base_url: str = "https://paper-api.alpaca.markets/v2"


@dataclass(frozen=True)
class AlphaVantageConfig:
    """Alpha Vantage API credentials."""

    api_key: str = field(default="", repr=False)


@dataclass(frozen=True)
class ModelConfig:
    """Date ranges, portfolio, and backtesting parameters."""

    historical_start_date: str = "2020-01-01"
    historical_end_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).date().isoformat()
    )
    initial_cash: float = 1000.0
    min_backtest_periods: int = 5
    start_index_percentage_backtesting: float = 0.5
    transaction_cost_rate: float = 0.001
    risk_free_rate: float = 0.0
    backtest_start_date: str | None = None


@dataclass(frozen=True)
class PathsConfig:
    """File-system paths."""

    input_data_dir: str = "data/input"


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection settings."""

    url: str = field(
        default="postgresql://investment_tool:investment_tool@localhost:5432/investment_tool",
        repr=False,
    )


@dataclass(frozen=True)
class RedisConfig:
    """Redis connection settings."""

    url: str = field(default="redis://localhost:6379/0", repr=False)


@dataclass(frozen=True)
class JWTConfig:
    """JSON Web Token settings."""

    secret_key: str = field(default="CHANGE-ME-IN-PRODUCTION", repr=False)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_minutes: int = 10080  # 7 days


@dataclass(frozen=True)
class AppConfig:
    """Top-level typed configuration, assembled from all sources.

    Construct via :func:`load_config` rather than directly.
    """

    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    alpha_vantage: AlphaVantageConfig = field(default_factory=AlphaVantageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    asset_config_file: str = DEFAULT_ASSET_CONFIG_FILE
    paths: PathsConfig = field(default_factory=PathsConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    jwt: JWTConfig = field(default_factory=JWTConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def get_default_config() -> str:
    """Load the default configuration from the bundled config.ini file.

    Returns
    -------
    str
        Contents of the default configuration file.
    """
    # Use modern importlib.resources API (Python 3.9+)
    files = importlib.resources.files("investment_tool")
    config_file = files.joinpath("config.ini")
    return config_file.read_text(encoding="utf-8")


def _build_app_config(cp: configparser.ConfigParser) -> AppConfig:
    """Build a typed ``AppConfig`` from a populated ``ConfigParser``."""
    return AppConfig(
        alpaca=AlpacaConfig(
            api_key=cp.get("ALPACA", "alpaca_api_key", fallback=""),
            api_secret=cp.get("ALPACA", "alpaca_api_secret", fallback=""),
            base_url=cp.get(
                "ALPACA",
                "alpaca_base_url",
                fallback="https://paper-api.alpaca.markets/v2",
            ),
        ),
        alpha_vantage=AlphaVantageConfig(
            api_key=cp.get("ALPHA-VANTAGE", "alpha_vantage_api_key", fallback=""),
        ),
        model=ModelConfig(
            historical_start_date=cp.get(
                "MODEL",
                "historical_start_date",
                fallback="2020-01-01",
            ),
            historical_end_date=cp.get(
                "MODEL",
                "historical_end_date",
                fallback=ModelConfig().historical_end_date,
            ),
            initial_cash=cp.getfloat("MODEL", "initial_cash", fallback=1000.0),
            min_backtest_periods=cp.getint(
                "MODEL",
                "min_backtest_periods",
                fallback=5,
            ),
            start_index_percentage_backtesting=cp.getfloat(
                "MODEL",
                "start_index_percentage_backtesting",
                fallback=0.5,
            ),
            transaction_cost_rate=cp.getfloat(
                "MODEL",
                "transaction_cost_rate",
                fallback=0.001,
            ),
            risk_free_rate=cp.getfloat("MODEL", "risk_free_rate", fallback=0.0),
            backtest_start_date=cp.get("MODEL", "backtest_start_date", fallback="")
            or None,
        ),
        paths=PathsConfig(
            input_data_dir=cp.get("PATHS", "input_data_dir", fallback="data/input"),
        ),
        database=DatabaseConfig(
            url=cp.get(
                "DATABASE",
                "url",
                fallback="postgresql://investment_tool:investment_tool@localhost:5432/investment_tool",
            ),
        ),
        redis=RedisConfig(
            url=cp.get("REDIS", "url", fallback="redis://localhost:6379/0"),
        ),
        jwt=JWTConfig(
            secret_key=cp.get("JWT", "secret_key", fallback="CHANGE-ME-IN-PRODUCTION"),
            algorithm=cp.get("JWT", "algorithm", fallback="HS256"),
            access_token_expire_minutes=cp.getint(
                "JWT", "access_token_expire_minutes", fallback=30
            ),
            refresh_token_expire_minutes=cp.getint(
                "JWT", "refresh_token_expire_minutes", fallback=10080
            ),
        ),
    )


def load_config(
    config_ini_path: Path | None = None,
    local_ini_path: Path | None = None,
    *,
    apply_env_overrides: bool = True,
) -> AppConfig:
    """Build a typed :class:`AppConfig` from config files and environment.

    Parameters
    ----------
    config_ini_path : Path | None
        Path to the base ``config.ini``.  ``None`` uses the bundled default.
    local_ini_path : Path | None
        Path to ``config.local.ini`` overrides.  ``None`` uses the standard
        location (``<project_root>/config.local.ini``).
    apply_env_overrides : bool
        Whether to overlay environment variables on top of file values.

    Returns
    -------
    AppConfig
        Frozen, typed configuration.
    """
    cp = configparser.ConfigParser()

    # Layer 1 — bundled defaults (or caller-supplied path)
    if config_ini_path is not None:
        cp.read([config_ini_path])
    else:
        cp.read_string(get_default_config())

    # Layer 2 — local overrides
    if local_ini_path is not None:
        cp.read([local_ini_path])
    else:
        default_local = Path(__file__).parent.parent.parent / "config.local.ini"
        cp.read([default_local])

    # Layer 3 — environment variables
    if apply_env_overrides:
        _apply_env_overrides(cp)

    return _build_app_config(cp)


def _apply_env_overrides(cp: configparser.ConfigParser) -> None:
    """Overlay literal environment values, escaping ConfigParser interpolation."""
    if "ALPHA_VANTAGE_API_KEY" in os.environ:
        cp.set(
            "ALPHA-VANTAGE",
            "alpha_vantage_api_key",
            os.environ["ALPHA_VANTAGE_API_KEY"].replace("%", "%%"),
        )
    if "ALPACA_API_KEY" in os.environ:
        cp.set(
            "ALPACA", "alpaca_api_key", os.environ["ALPACA_API_KEY"].replace("%", "%%")
        )
    if "ALPACA_API_SECRET" in os.environ:
        cp.set(
            "ALPACA",
            "alpaca_api_secret",
            os.environ["ALPACA_API_SECRET"].replace("%", "%%"),
        )
    # Database, Redis, JWT
    if "DATABASE_URL" in os.environ:
        if not cp.has_section("DATABASE"):
            cp.add_section("DATABASE")
        cp.set("DATABASE", "url", os.environ["DATABASE_URL"].replace("%", "%%"))
    if "REDIS_URL" in os.environ:
        if not cp.has_section("REDIS"):
            cp.add_section("REDIS")
        cp.set("REDIS", "url", os.environ["REDIS_URL"].replace("%", "%%"))
    if "JWT_SECRET_KEY" in os.environ:
        if not cp.has_section("JWT"):
            cp.add_section("JWT")
        cp.set("JWT", "secret_key", os.environ["JWT_SECRET_KEY"].replace("%", "%%"))


# ---------------------------------------------------------------------------
# Lazy default singleton
# ---------------------------------------------------------------------------

_default_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the lazy default :class:`AppConfig`.

    On first call this reads config files and environment variables via
    :func:`load_config`.  Subsequent calls return the cached result.

    Prefer accepting an explicit ``cfg`` parameter in library code and
    use this function only as a convenience fallback.
    """
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config


def reset_config() -> None:
    """Clear the cached default config (useful for tests).

    The next call to :func:`get_config` will re-read from disk/env.
    """
    global _default_config
    _default_config = None


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated)
# ---------------------------------------------------------------------------


class _LazyAppConfig:
    """Descriptor that defers ``app_config`` access to :func:`get_config`.

    This lets existing code that does ``from investment_tool.config import
    app_config`` keep working, but the disk I/O only happens on first
    *use*, not on import.

    .. deprecated::
        Use :func:`get_config` or, better, accept an explicit ``cfg``
        parameter.
    """

    def __repr__(self) -> str:
        return repr(get_config())

    def __getattr__(self, name: str) -> object:
        warnings.warn(
            "Accessing the module-level 'app_config' singleton is deprecated. "
            "Use get_config() or pass an explicit AppConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(get_config(), name)


app_config: AppConfig = _LazyAppConfig()  # type: ignore[assignment]
