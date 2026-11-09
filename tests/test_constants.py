"""Unit tests for constants and enumerations."""

from __future__ import annotations

import pytest

from investment_tool.constants import (
    DataSource,
    ExecutionMode,
    Frequency,
    StockFields,
)


class TestStringEnum:
    """Tests for StringEnum base class."""

    def test_str_returns_value(self) -> None:
        """Test __str__ returns the enum value."""
        assert str(DataSource.ALPACA) == "alpaca"
        assert str(DataSource.LOCAL) == "local"

    def test_string_comparison(self) -> None:
        """Test enum can be compared to string directly."""
        # StringEnum inherits from str, so comparison works
        assert DataSource.ALPACA == "alpaca"
        assert DataSource.LOCAL == "local"

    def test_list_method(self) -> None:
        """Test list() class method returns all values."""
        values = DataSource.list()
        assert isinstance(values, list)
        assert "alpaca" in values
        assert "local" in values

    def test_list_returns_strings(self) -> None:
        """Test list() returns strings, not enum members."""
        values = DataSource.list()
        for v in values:
            assert isinstance(v, str)


class TestDataSource:
    """Tests for DataSource enum."""

    def test_alpaca_value(self) -> None:
        """Test ALPACA enum value."""
        assert DataSource.ALPACA.value == "alpaca"

    def test_local_value(self) -> None:
        """Test LOCAL enum value."""
        assert DataSource.LOCAL.value == "local"

    def test_all_values_unique(self) -> None:
        """Test all enum values are unique."""
        values = [member.value for member in DataSource]
        assert len(values) == len(set(values))


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_investing_mode(self) -> None:
        """Test INVESTING mode exists."""
        assert hasattr(ExecutionMode, "INVESTING")

    def test_backtesting_mode(self) -> None:
        """Test BACKTESTING mode exists."""
        assert hasattr(ExecutionMode, "BACKTESTING")

    def test_mode_values(self) -> None:
        """Test mode values are strings."""
        assert isinstance(ExecutionMode.INVESTING.value, str)
        assert isinstance(ExecutionMode.BACKTESTING.value, str)


class TestFrequency:
    """Tests for Frequency enum."""

    def test_daily_frequency(self) -> None:
        """Test DAILY frequency exists."""
        assert hasattr(Frequency, "DAILY")
        assert Frequency.DAILY.value == "daily"

    def test_weekly_frequency(self) -> None:
        """Test WEEKLY frequency exists."""
        assert hasattr(Frequency, "WEEKLY")
        assert Frequency.WEEKLY.value == "weekly"

    def test_monthly_frequency(self) -> None:
        """Test MONTHLY frequency exists."""
        assert hasattr(Frequency, "MONTHLY")
        assert Frequency.MONTHLY.value == "monthly"

    def test_frequency_list(self) -> None:
        """Test Frequency.list() returns all frequencies."""
        freq_list = Frequency.list()
        assert "daily" in freq_list
        assert "weekly" in freq_list
        assert "monthly" in freq_list


class TestStockFields:
    """Tests for StockFields constants."""

    def test_ohlcv_fields_exist(self) -> None:
        """Test OHLCV field names exist."""
        assert hasattr(StockFields, "OPEN")
        assert hasattr(StockFields, "HIGH")
        assert hasattr(StockFields, "LOW")
        assert hasattr(StockFields, "CLOSE")
        assert hasattr(StockFields, "VOLUME")

    def test_technical_indicator_fields(self) -> None:
        """Test technical indicator fields exist."""
        assert hasattr(StockFields, "RETURNS")
        assert hasattr(StockFields, "LOG_RETURNS")
        assert hasattr(StockFields, "VOLATILITY")
        assert hasattr(StockFields, "RSI")

    def test_moving_average_field(self) -> None:
        """Test moving average base field exists."""
        assert hasattr(StockFields, "MOVING_AVERAGE")

    def test_bollinger_band_fields(self) -> None:
        """Test Bollinger Band fields exist."""
        assert hasattr(StockFields, "UPPER_BAND")
        assert hasattr(StockFields, "MIDDLE_BAND")
        assert hasattr(StockFields, "LOWER_BAND")

    def test_field_values_are_strings(self) -> None:
        """Test field values are lowercase strings."""
        assert StockFields.OPEN == "open"
        assert StockFields.CLOSE == "close"
        assert StockFields.RETURNS == "returns"


class TestEnumMembership:
    """Tests for enum membership checks."""

    def test_data_source_membership(self) -> None:
        """Test membership check for DataSource."""
        assert "alpaca" in DataSource.list()
        assert "invalid" not in DataSource.list()

    def test_frequency_membership(self) -> None:
        """Test membership check for Frequency."""
        assert "daily" in Frequency.list()
        assert "hourly" not in Frequency.list()

    def test_iteration(self) -> None:
        """Test that enums are iterable."""
        sources = list(DataSource)
        assert len(sources) >= 2
        assert all(isinstance(s, DataSource) for s in sources)


class TestEnumFromValue:
    """Tests for creating enums from values."""

    def test_create_from_value(self) -> None:
        """Test creating enum from string value."""
        source = DataSource("alpaca")
        assert source == DataSource.ALPACA

    def test_invalid_value_raises(self) -> None:
        """Test invalid value raises ValueError."""
        with pytest.raises(ValueError):
            DataSource("invalid_source")

    def test_case_sensitive(self) -> None:
        """Test that enum creation is case-sensitive."""
        with pytest.raises(ValueError):
            DataSource("ALPACA")  # Should be lowercase
