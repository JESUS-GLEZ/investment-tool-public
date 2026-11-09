"""Unit tests for configuration management (config.py).

Covers:
- ``load_config`` with default and custom paths, env overrides
- ``get_config`` lazy caching behaviour
- ``reset_config`` cache clearing
- ``_LazyAppConfig`` backward-compatibility proxy and deprecation warning
- ``_build_app_config`` helper
"""

from __future__ import annotations

import configparser
import textwrap
import warnings
from pathlib import Path

import pytest

from investment_tool.config import (
    AppConfig,
    AlpacaConfig,
    AlphaVantageConfig,
    DatabaseConfig,
    JWTConfig,
    ModelConfig,
    PathsConfig,
    RedisConfig,
    _LazyAppConfig,
    _build_app_config,
    get_config,
    get_default_config,
    load_config,
    reset_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def test_config_repr_does_not_disclose_credentials() -> None:
    cfg = AppConfig(
        alpaca=AlpacaConfig(api_key="synthetic-key", api_secret="synthetic-secret"),
        alpha_vantage=AlphaVantageConfig(api_key="synthetic-av-key"),
        database=DatabaseConfig(url="postgresql://user:synthetic-db-pass@localhost/db"),
        redis=RedisConfig(url="redis://:synthetic-redis-pass@localhost/0"),
        jwt=JWTConfig(secret_key="synthetic-jwt-secret"),
    )
    assert "synthetic" not in repr(cfg)
    assert cfg.alpaca.api_key == "synthetic-key"
    assert cfg.jwt.secret_key == "synthetic-jwt-secret"


def test_environment_percent_characters_are_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = {
        "DATABASE_URL": "postgresql://user:synthetic%40password@localhost/db",
        "REDIS_URL": "redis://:synthetic%25password@localhost/0",
        "JWT_SECRET_KEY": "synthetic-signing-secret-with-literal-%",
        "ALPACA_API_KEY": "synthetic-key-%",
        "ALPACA_API_SECRET": "synthetic-secret-%",
        "ALPHA_VANTAGE_API_KEY": "synthetic-av-%",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    cfg = load_config(local_ini_path=tmp_path / "nonexistent.ini")
    assert cfg.database.url == values["DATABASE_URL"]
    assert cfg.redis.url == values["REDIS_URL"]
    assert cfg.jwt.secret_key == values["JWT_SECRET_KEY"]
    assert cfg.alpaca.api_key == values["ALPACA_API_KEY"]
    assert cfg.alpaca.api_secret == values["ALPACA_API_SECRET"]
    assert cfg.alpha_vantage.api_key == values["ALPHA_VANTAGE_API_KEY"]


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:  # type: ignore[misc]
    """Ensure the lazy singleton is cleared before and after every test."""
    reset_config()
    yield  # type: ignore[misc]
    reset_config()


@pytest.fixture()
def sample_ini(tmp_path: Path) -> Path:
    """Write a minimal ``config.ini`` to *tmp_path* and return its path."""
    ini = tmp_path / "config.ini"
    ini.write_text(
        textwrap.dedent(
            """\
            [ALPHA-VANTAGE]
            alpha_vantage_api_key = av-test-key

            [ALPACA]
            alpaca_api_key = alp-key
            alpaca_api_secret = alp-secret
            alpaca_base_url = https://custom.alpaca.example/v2

            [MODEL]
            historical_start_date = 2021-06-01
            historical_end_date   = 2025-12-31
            initial_cash          = 5000.0
            min_backtest_periods  = 10
            start_index_percentage_backtesting = 0.3
            transaction_cost_rate = 0.002

            [PATHS]
            input_data_dir = /tmp/test-data
        """
        ),
        encoding="utf-8",
    )
    return ini


@pytest.fixture()
def empty_local_ini(tmp_path: Path) -> Path:
    """Return the path to a non-existent local override file."""
    return tmp_path / "config.local.ini"


# ---------------------------------------------------------------------------
# _build_app_config
# ---------------------------------------------------------------------------


class TestBuildAppConfig:
    """Tests for ``_build_app_config``."""

    def test_builds_from_configparser(self, sample_ini: Path) -> None:
        cp = configparser.ConfigParser()
        cp.read([sample_ini])
        cfg = _build_app_config(cp)

        assert isinstance(cfg, AppConfig)
        assert cfg.alpaca.api_key == "alp-key"
        assert cfg.alpaca.api_secret == "alp-secret"
        assert cfg.alpaca.base_url == "https://custom.alpaca.example/v2"
        assert cfg.alpha_vantage.api_key == "av-test-key"
        assert cfg.model.historical_start_date == "2021-06-01"
        assert cfg.model.historical_end_date == "2025-12-31"
        assert cfg.model.initial_cash == 5000.0
        assert cfg.model.min_backtest_periods == 10
        assert cfg.model.start_index_percentage_backtesting == 0.3
        assert cfg.model.transaction_cost_rate == 0.002
        assert cfg.asset_config_file == "example_asset_universe.json"
        assert cfg.paths.input_data_dir == "/tmp/test-data"

    def test_fallback_defaults_on_empty_configparser(self) -> None:
        """All fields fall back to dataclass defaults when ConfigParser is empty."""
        cp = configparser.ConfigParser()
        cfg = _build_app_config(cp)

        assert cfg.alpaca.api_key == ""
        assert cfg.alpha_vantage.api_key == ""
        assert cfg.model.initial_cash == 1000.0
        assert cfg.asset_config_file == "example_asset_universe.json"
        assert cfg.paths.input_data_dir == "data/input"


# ---------------------------------------------------------------------------
# get_default_config
# ---------------------------------------------------------------------------


class TestGetDefaultConfig:
    """Tests for ``get_default_config``."""

    def test_returns_nonempty_string(self) -> None:
        raw = get_default_config()
        assert isinstance(raw, str)
        assert len(raw) > 0

    def test_parseable_ini(self) -> None:
        """The bundled config.ini must be a valid INI file."""
        cp = configparser.ConfigParser()
        cp.read_string(get_default_config())
        assert "ALPACA" in cp.sections()
        assert "MODEL" in cp.sections()


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for ``load_config``."""

    def test_loads_bundled_defaults(self, empty_local_ini: Path) -> None:
        """Without arguments, returns a valid AppConfig from bundled config.ini."""
        cfg = load_config(local_ini_path=empty_local_ini, apply_env_overrides=False)
        assert isinstance(cfg, AppConfig)
        # Bundled config.ini has these values
        assert cfg.model.historical_start_date == "2020-01-01"
        assert cfg.model.initial_cash == 1000.0

    def test_custom_config_ini_path(
        self, sample_ini: Path, empty_local_ini: Path
    ) -> None:
        cfg = load_config(
            config_ini_path=sample_ini,
            local_ini_path=empty_local_ini,
            apply_env_overrides=False,
        )
        assert cfg.alpaca.api_key == "alp-key"
        assert cfg.model.initial_cash == 5000.0

    def test_local_ini_overrides_base(self, tmp_path: Path) -> None:
        """A local override file should take precedence over the base config."""
        base = tmp_path / "config.ini"
        base.write_text(
            textwrap.dedent(
                """\
                [MODEL]
                initial_cash = 1000.0
            """
            ),
            encoding="utf-8",
        )
        local = tmp_path / "config.local.ini"
        local.write_text(
            textwrap.dedent(
                """\
                [MODEL]
                initial_cash = 9999.0
            """
            ),
            encoding="utf-8",
        )
        cfg = load_config(
            config_ini_path=base,
            local_ini_path=local,
            apply_env_overrides=False,
        )
        assert cfg.model.initial_cash == 9999.0

    def test_env_override_alpaca_key(
        self, empty_local_ini: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALPACA_API_KEY", "env-alpaca-key")
        cfg = load_config(
            local_ini_path=empty_local_ini,
            apply_env_overrides=True,
        )
        assert cfg.alpaca.api_key == "env-alpaca-key"

    def test_env_override_alpaca_secret(
        self, empty_local_ini: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALPACA_API_SECRET", "env-alpaca-secret")
        cfg = load_config(
            local_ini_path=empty_local_ini,
            apply_env_overrides=True,
        )
        assert cfg.alpaca.api_secret == "env-alpaca-secret"

    def test_env_override_alpha_vantage_key(
        self, empty_local_ini: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "env-av-key")
        cfg = load_config(
            local_ini_path=empty_local_ini,
            apply_env_overrides=True,
        )
        assert cfg.alpha_vantage.api_key == "env-av-key"

    def test_env_overrides_disabled(
        self, empty_local_ini: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``apply_env_overrides=False``, environment variables are ignored."""
        monkeypatch.setenv("ALPACA_API_KEY", "should-be-ignored")
        cfg = load_config(
            local_ini_path=empty_local_ini,
            apply_env_overrides=False,
        )
        # Bundled config.ini has an empty key
        assert cfg.alpaca.api_key == ""

    def test_returns_frozen_dataclass(self, empty_local_ini: Path) -> None:
        cfg = load_config(local_ini_path=empty_local_ini, apply_env_overrides=False)
        with pytest.raises(AttributeError):
            cfg.alpaca = AlpacaConfig(api_key="nope")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_config / reset_config  (lazy singleton)
# ---------------------------------------------------------------------------


class TestGetConfigSingleton:
    """Tests for ``get_config`` and ``reset_config``."""

    def test_returns_app_config(self) -> None:
        cfg = get_config()
        assert isinstance(cfg, AppConfig)

    def test_caches_result(self) -> None:
        """Successive calls return the exact same object."""
        first = get_config()
        second = get_config()
        assert first is second

    def test_reset_clears_cache(self) -> None:
        """After ``reset_config()``, ``get_config()`` builds a new object."""
        first = get_config()
        reset_config()
        second = get_config()
        # Same *values* (both use bundled defaults) but different *objects*
        assert first is not second
        assert first == second

    def test_no_disk_io_on_import(self) -> None:
        """Importing the module should NOT eagerly build a config.

        The ``_default_config`` module variable starts as ``None``.
        """
        import investment_tool.config as cfg_mod

        # After reset_config() (called by autouse fixture) the cache is None
        assert cfg_mod._default_config is None


# ---------------------------------------------------------------------------
# _LazyAppConfig  (backward-compatibility proxy)
# ---------------------------------------------------------------------------


class TestLazyAppConfigProxy:
    """Tests for the ``_LazyAppConfig`` backward-compatibility proxy."""

    def test_attribute_access_returns_correct_value(self) -> None:
        proxy = _LazyAppConfig()
        # Should delegate to get_config().model
        model = proxy.model  # type: ignore[attr-defined]
        assert isinstance(model, ModelConfig)

    def test_emits_deprecation_warning(self) -> None:
        proxy = _LazyAppConfig()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = proxy.alpaca  # type: ignore[attr-defined]

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "app_config" in str(w[0].message)
        assert "get_config()" in str(w[0].message)

    def test_repr_delegates_to_real_config(self) -> None:
        proxy = _LazyAppConfig()
        assert repr(proxy) == repr(get_config())

    def test_module_level_app_config_is_proxy(self) -> None:
        """The module-level ``app_config`` should be a ``_LazyAppConfig``."""
        from investment_tool.config import app_config

        assert isinstance(app_config, _LazyAppConfig)

    def test_module_level_app_config_emits_warning(self) -> None:
        from investment_tool.config import app_config

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = app_config.alpaca  # type: ignore[attr-defined]

        assert any(issubclass(x.category, DeprecationWarning) for x in w)


# ---------------------------------------------------------------------------
# AppConfig dataclass properties
# ---------------------------------------------------------------------------


class TestAppConfigDataclass:
    """Sanity checks on the typed config dataclasses."""

    def test_default_construction(self) -> None:
        cfg = AppConfig()
        assert cfg.alpaca == AlpacaConfig()
        assert cfg.alpha_vantage == AlphaVantageConfig()
        assert cfg.model == ModelConfig()
        assert cfg.asset_config_file == "example_asset_universe.json"
        assert cfg.paths == PathsConfig()

    def test_frozen(self) -> None:
        cfg = AppConfig()
        with pytest.raises(AttributeError):
            cfg.alpaca = AlpacaConfig(api_key="x")  # type: ignore[misc]

    def test_equality(self) -> None:
        a = AppConfig()
        b = AppConfig()
        assert a == b

    def test_asset_config_file(self) -> None:
        cfg = AppConfig(asset_config_file="custom_example.json")
        assert cfg.asset_config_file == "custom_example.json"


def test_default_cutoff_is_evaluated_when_config_is_created(monkeypatch):
    from datetime import datetime, timezone
    import investment_tool.config as config

    class Clock(datetime):
        current = datetime(2026, 10, 27, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(config, "datetime", Clock)
    assert ModelConfig().historical_end_date == "2026-10-27"
    Clock.current = datetime(2027, 1, 1, tzinfo=timezone.utc)
    assert ModelConfig().historical_end_date == "2027-01-01"
    assert (
        _build_app_config(configparser.ConfigParser()).model.historical_end_date
        == "2027-01-01"
    )
    assert (
        ModelConfig(historical_end_date="2024-06-30").historical_end_date
        == "2024-06-30"
    )
