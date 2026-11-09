"""Provider-boundary normalization, persistence and backtest timeline parity."""

import pickle
from datetime import date

import pandas as pd
import pytest

from investment_tool.config import AppConfig, ModelConfig
from investment_tool.constants import DataSource, ExecutionMode, Frequency
from investment_tool.data.assembly.model_data_factory import ModelDataFactory
from investment_tool.data.validation import ValidationLevel
from investment_tool.data.repositories.base_repository import (
    BaseDataRepository,
    DataResult,
)
from investment_tool.data.repositories.local_repository import LocalRepository
from investment_tool.data.repositories.repository_factory import RepositoryFactory
from investment_tool.data.repositories._alpaca_client import AlpacaDataFetcher
from investment_tool.data.upgrade_time_data import (
    save_upgraded_data,
    upgrade_legacy_alpaca,
)
from investment_tool.domain.backtesting.backtester import Backtester
from investment_tool.domain.market_time import normalize_bars
from investment_tool.output.metrics import MetricsCalculator
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)


def monthly_bars(zone="UTC"):
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "high": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "close": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "volume": [100.0] * 6,
        },
        index=pd.date_range("2024-01-01", periods=6, freq="MS", tz=zone),
    )


class FakeProvider(BaseDataRepository):
    def __init__(self, convention):
        super().__init__(Frequency.MONTHLY)
        self.convention = convention

    @property
    def source_name(self):
        return f"Fake-{self.convention}"

    def fetch_stock_data(self, symbols, start_date, end_date):
        raw = monthly_bars()
        if self.convention == "end":
            raw.index += pd.offsets.MonthBegin()
        return {
            "ABC": normalize_bars(
                raw,
                self.frequency,
                market_timezone="UTC",
                timestamp_convention=self.convention,
            )
        }

    def fetch_crypto_data(self, *args):
        return {}

    def fetch_news_data(self, *args):
        return {}


class ObservingStrategy(BaseInvestmentStrategy):
    def __init__(self, instance, stocks):
        self.observed = []
        super().__init__(instance, stocks, None)

    def run_strategy(self, progress_callback=None):
        period = self._instance_ref.current_trading_period
        history = self.stocks[0].data
        self.observed.append((period.decision_time, history.index.tolist()))
        assert not history.empty
        assert (history.available_at <= pd.Timestamp(period.decision_time)).all()
        return pd.Series({"ABC": 1.0})


def test_two_providers_generate_identical_decisions_valuations_and_periods(monkeypatch):
    outputs = []
    histories = []
    for convention in ("start", "end"):
        monkeypatch.setattr(
            RepositoryFactory, "create", lambda **kwargs: FakeProvider(convention)
        )
        factory = ModelDataFactory(
            DataSource.ALPACA,
            ExecutionMode.BACKTESTING,
            Frequency.MONTHLY,
            0,
            validation_level=ValidationLevel.SKIP,
            cfg=AppConfig(
                model=ModelConfig(
                    historical_start_date="2024-01-01",
                    historical_end_date="2024-06-30",
                    min_backtest_periods=2,
                    start_index_percentage_backtesting=0.5,
                )
            ),
        )
        model = factory._create_model_data()
        initial_cash = model.portfolio.cash
        strategy = ObservingStrategy(model.instance, model.stocks)
        backtest = Backtester(model, strategy)
        backtest.run_backtest()
        metrics = MetricsCalculator(
            backtest.results, model.portfolio.history, Frequency.MONTHLY
        ).compute_all()
        assert metrics["initial_value"] == pytest.approx(initial_cash)
        assert metrics["total_return"] == pytest.approx(
            metrics["final_value"] / metrics["initial_value"] - 1
        )
        assert metrics["metric_definitions"]["annualized_return"] == "cagr"
        outputs.append(backtest.get_results())
        histories.append(strategy.observed)
    pd.testing.assert_frame_equal(outputs[0], outputs[1])
    assert histories[0] == histories[1]
    assert outputs[0].index.tolist() == list(
        pd.to_datetime(["2024-04-30", "2024-05-31", "2024-06-30"])
    )
    assert outputs[0].period_start.tolist() == [
        "2024-04-01",
        "2024-05-01",
        "2024-06-01",
    ]
    assert outputs[0].date_basis.eq("period_end_label").all()


def test_backtest_excludes_delayed_bar_until_its_actual_availability(monkeypatch):
    class DelayedProvider(FakeProvider):
        def fetch_stock_data(self, *args):
            result = super().fetch_stock_data(*args)
            result["ABC"].loc[pd.Timestamp("2024-03-01"), "available_at"] = (
                pd.Timestamp("2024-04-15T00:00:00Z")
            )
            return result

    monkeypatch.setattr(
        RepositoryFactory, "create", lambda **kwargs: DelayedProvider("start")
    )
    model = ModelDataFactory(
        DataSource.ALPACA,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        validation_level=ValidationLevel.SKIP,
        cfg=AppConfig(
            model=ModelConfig(
                historical_start_date="2024-01-01",
                historical_end_date="2024-06-30",
                min_backtest_periods=2,
                start_index_percentage_backtesting=0.5,
            )
        ),
    )._create_model_data()
    strategy = ObservingStrategy(model.instance, model.stocks)
    Backtester(model, strategy).run_backtest()
    first_cutoff, first_history = strategy.observed[0]
    assert first_cutoff == pd.Timestamp("2024-04-01T00:00:00Z")
    assert pd.Timestamp("2024-03-01") not in first_history
    assert pd.Timestamp("2024-03-01") in strategy.observed[1][1]


def test_repository_rejects_untimed_provider_before_fetching_news():
    class UntimedProvider(FakeProvider):
        def fetch_stock_data(self, *args):
            return {"ABC": monthly_bars()}

        def fetch_news_data(self, *args):
            pytest.fail("Invalid timing must fail before slow news retrieval")

    with pytest.raises(ValueError, match="Missing bar time contract"):
        UntimedProvider("start").fetch_all(["ABC"], [], "2024-01-01", "2024-06-30")


@pytest.mark.parametrize("suffix", [".pkl", ".json"])
def test_versioned_local_round_trip_preserves_timing_and_metadata(tmp_path, suffix):
    canonical = normalize_bars(
        monthly_bars("America/New_York"),
        Frequency.MONTHLY,
        market_timezone="America/New_York",
    )
    data = DataResult(
        stock_dict={"ABC": canonical},
        metadata={"time_contract_version": 1, "frequency": "monthly", "source": "Fake"},
        execution_dict={
            "2024-04-01": {"timestamp": "2024-04-01T13:30:00Z", "crypto_prices": {}}
        },
    )
    path = tmp_path / f"snapshot{suffix}"
    save_upgraded_data(data.to_dict(), path)
    loaded = LocalRepository(Frequency.MONTHLY, path).fetch_all(
        ["ABC"], [], "2024-01-01", "2024-06-30"
    )
    pd.testing.assert_frame_equal(loaded.stock_dict["ABC"], canonical, check_freq=False)
    assert loaded.metadata["source"] == "Fake"
    assert loaded.metadata["time_contract_version"] == 1
    assert loaded.execution_dict == data.execution_dict
    with pytest.raises(FileExistsError):
        save_upgraded_data(data.to_dict(), path)


def test_unversioned_and_wrong_frequency_local_files_are_rejected(tmp_path):
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps({"stock_dict": {"ABC": monthly_bars()}}))
    with pytest.raises(ValueError, match="upgrade_time_data"):
        LocalRepository(Frequency.MONTHLY, path).fetch_all(
            ["ABC"], [], "2024-01-01", "2024-06-30"
        )
    path.write_bytes(
        pickle.dumps(
            DataResult(
                stock_dict={
                    "ABC": normalize_bars(
                        monthly_bars(), Frequency.MONTHLY, market_timezone="UTC"
                    )
                },
                metadata={"time_contract_version": 1, "frequency": "monthly"},
            ).to_dict()
        )
    )
    with pytest.raises(ValueError, match="frequency"):
        LocalRepository(Frequency.DAILY, path)._load_data()


def test_explicit_legacy_upgrade_restores_crypto_boundary_and_keeps_raw_news():
    raw = monthly_bars()
    raw.index -= pd.Timedelta(hours=1)
    original = raw.copy()
    news = {"ABC": {("2024-01-01", "2024-02-01"): [{"headline": "unchanged"}]}}
    upgraded = upgrade_legacy_alpaca(
        {"crypto_dict": {"BTC/USD": raw}, "stock_dict": {}, "news_dict": news},
        Frequency.MONTHLY,
        start_date="2024-01-01",
        end_date="2024-06-01",
        retrieved_at="2024-06-15T00:00:00Z",
    )
    frame = upgraded["crypto_dict"]["BTC/USD"]
    assert frame.index[0] == pd.Timestamp("2024-01-01")
    assert frame.is_complete.tolist() == [True] * 5 + [False]
    assert upgraded["news_dict"] == news
    pd.testing.assert_frame_equal(raw, original)
    with pytest.raises(ValueError, match="already versioned"):
        upgrade_legacy_alpaca(
            upgraded,
            Frequency.MONTHLY,
            start_date="2024-01-01",
            end_date="2024-06-01",
            retrieved_at="2024-06-15T00:00:00Z",
        )


def test_investing_uses_latest_completed_month_not_an_old_historical_anchor(
    monkeypatch,
):
    monkeypatch.setattr(
        RepositoryFactory, "create", lambda **kwargs: FakeProvider("start")
    )
    factory = ModelDataFactory(
        DataSource.ALPACA,
        ExecutionMode.INVESTING,
        Frequency.MONTHLY,
        0,
        validation_level=ValidationLevel.SKIP,
        cfg=AppConfig(
            model=ModelConfig(
                historical_start_date="2024-01-01", historical_end_date="2024-06-30"
            )
        ),
    )
    model = factory._create_model_data()
    strategy = ObservingStrategy(model.instance, model.stocks)
    assert strategy.stocks[0].data.index[-1] == pd.Timestamp("2024-06-01")
    assert model.instance.current_trading_period.decision_time.tzinfo is not None


def test_alpaca_adapter_preserves_month_and_declares_timezone_without_offsets(
    monkeypatch,
):
    from types import SimpleNamespace

    client = object.__new__(AlpacaDataFetcher)
    client.stock_client = SimpleNamespace(get_stock_bars=object())
    client.crypto_client = SimpleNamespace(get_crypto_bars=object())
    requests = []

    def fetch(method, **kwargs):
        requests.append(kwargs["request_params"].to_request_fields())
        frame = monthly_bars("America/New_York")
        frame["symbol"] = "ABC"
        return SimpleNamespace(df=frame.set_index("symbol", append=True))

    monkeypatch.setattr(client, "_handle_api_request", fetch)
    result = client.get_historical_data(
        ["ABC"], "2024-01-01", "2024-06-30", Frequency.MONTHLY
    )["ABC"]
    assert result.index.tolist() == list(
        pd.date_range("2024-01-01", periods=6, freq="MS")
    )
    assert result.market_timezone.eq("America/New_York").all()
    assert requests[0]["end"] == "2024-07-01T04:00:00+00:00"


def test_symbol_identity_date_maps_stocks_without_changing_history_or_crypto(
    monkeypatch,
):
    from types import SimpleNamespace

    client = object.__new__(AlpacaDataFetcher)
    client.stock_client = SimpleNamespace(get_stock_bars=object())
    client.crypto_client = SimpleNamespace(get_crypto_bars=object())
    requests = []

    def fetch(method, **kwargs):
        requests.append(kwargs["request_params"].to_request_fields())
        is_crypto = method is client.crypto_client.get_crypto_bars
        frame = monthly_bars("UTC" if is_crypto else "America/New_York")
        frame["symbol"] = "BTC/USD" if is_crypto else "ABC"
        return SimpleNamespace(df=frame.set_index("symbol", append=True))

    monkeypatch.setattr(client, "_handle_api_request", fetch)
    client.get_historical_data(
        ["ABC"],
        "2024-01-01",
        "2024-06-30",
        Frequency.MONTHLY,
        symbol_identity_date=date(2021, 1, 1),
    )
    assert requests[0]["asof"] == "2021-01-01"
    assert requests[0]["start"] == "2024-01-01T05:00:00+00:00"
    assert requests[0]["end"] == "2024-07-01T04:00:00+00:00"
    client.get_historical_crypto_data(
        ["BTC/USD"], "2024-01-01", "2024-06-30", Frequency.MONTHLY
    )
    assert "asof" not in requests[1]


@pytest.mark.parametrize("stored", [None, "2026-10-06", "2021-01-01"])
def test_local_snapshot_must_match_the_requested_symbol_identity(tmp_path, stored):
    path = tmp_path / "prices.pkl"
    data = {
        "stock_dict": {},
        "metadata": {
            "time_contract_version": 1,
            "frequency": "monthly",
            "ticker_identity_asof": stored,
        },
    }
    path.write_bytes(pickle.dumps(data))
    repo = LocalRepository(
        Frequency.MONTHLY, path, symbol_identity_date=date(2021, 1, 1)
    )
    if stored == "2021-01-01":
        assert repo._load_data().metadata["ticker_identity_asof"] == stored
    else:
        with pytest.raises(ValueError, match="ticker identity date"):
            repo._load_data()


def test_non_us_execution_in_previous_utc_month_uses_the_same_engine(monkeypatch):
    class OtherMarket(FakeProvider):
        def fetch_stock_data(self, *args):
            return {
                "ABC": normalize_bars(
                    monthly_bars("Pacific/Auckland"),
                    self.frequency,
                    market_timezone="Pacific/Auckland",
                )
            }

        def fetch_execution_data(self, *args):
            return {
                label.date().isoformat(): {
                    "timestamp": (label + pd.DateOffset(hours=9, minutes=30))
                    .tz_convert("UTC")
                    .isoformat(),
                    "crypto_prices": {},
                }
                for label in pd.date_range(
                    "2024-01-01", periods=6, freq="MS", tz="Pacific/Auckland"
                )
            }

    monkeypatch.setattr(
        RepositoryFactory, "create", lambda **kwargs: OtherMarket("start")
    )
    factory = ModelDataFactory(
        DataSource.ALPACA,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        validation_level=ValidationLevel.SKIP,
        cfg=AppConfig(
            model=ModelConfig(
                historical_start_date="2024-01-01",
                historical_end_date="2024-06-30",
                min_backtest_periods=2,
            )
        ),
    )
    model = factory._create_model_data()
    may = next(p for p in model.periods if p.trading_date == pd.Timestamp("2024-05-01"))
    assert may.execution_time.month == 4
    assert may.execution_time.tzinfo is not None
    assert may.decision_time < may.execution_time


def test_dataset_info_reports_complete_period_ends_not_start_labels(
    tmp_path, monkeypatch
):
    from api.routers import defaults

    canonical = normalize_bars(monthly_bars(), Frequency.MONTHLY, market_timezone="UTC")
    canonical.loc[pd.Timestamp("2024-06-01"), "is_complete"] = False
    path = tmp_path / "prices.pkl"
    save_upgraded_data(
        DataResult(
            stock_dict={"ABC": canonical},
            metadata={"time_contract_version": 1, "frequency": "monthly"},
        ).to_dict(),
        path,
    )
    monkeypatch.setattr(defaults, "_INPUT_BASE", tmp_path)
    info = defaults.get_local_data_info(path.name)
    assert info["start_date"] == "2024-01-01"
    assert info["end_date"] == "2024-05-31"
    assert info["time_contract_version"] == 1
