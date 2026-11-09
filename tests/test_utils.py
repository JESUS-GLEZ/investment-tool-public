"""Unit tests for utility functions."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from investment_tool.utils import (
    _final_ewm_correlation,
    align_timestamps,
    build_exp_weighted_covariance_matrix,
    get_extension,
    nearest_psd,
    time_diff,
    validate_covariance_matrix,
)


class TestGetExtension:
    """Tests for get_extension function."""

    def test_simple_extension(self) -> None:
        """Test extracting simple extension."""
        assert get_extension("data.csv") == ".csv"
        assert get_extension("file.pkl") == ".pkl"
        assert get_extension("report.json") == ".json"

    def test_multiple_dots(self) -> None:
        """Test file with multiple dots."""
        assert get_extension("data.backup.csv") == ".csv"
        assert get_extension("file.v2.pkl") == ".pkl"

    def test_with_path(self) -> None:
        """Test with full file path."""
        assert get_extension("/path/to/data.csv") == ".csv"
        assert get_extension("./relative/file.json") == ".json"

    def test_no_extension(self) -> None:
        """Test file without extension."""
        assert get_extension("filename") == ""
        assert get_extension("/path/to/file") == ""

    def test_hidden_file(self) -> None:
        """Test hidden files (Unix-style)."""
        assert get_extension(".gitignore") == ""
        assert get_extension(".env.local") == ".local"


class TestTimeDiff:
    """Tests for time_diff function."""

    def test_exact_minutes(self) -> None:
        """Test exact minute difference."""
        start = datetime.datetime(2024, 1, 1, 10, 0, 0)
        end = datetime.datetime(2024, 1, 1, 10, 30, 0)
        assert time_diff(start, end) == 30

    def test_hours_to_minutes(self) -> None:
        """Test hour difference converted to minutes."""
        start = datetime.datetime(2024, 1, 1, 10, 0, 0)
        end = datetime.datetime(2024, 1, 1, 12, 0, 0)
        assert time_diff(start, end) == 120

    def test_negative_diff(self) -> None:
        """Test when end is before start (negative diff)."""
        start = datetime.datetime(2024, 1, 1, 12, 0, 0)
        end = datetime.datetime(2024, 1, 1, 10, 0, 0)
        assert time_diff(start, end) == -120

    def test_zero_diff(self) -> None:
        """Test same time."""
        same_time = datetime.datetime(2024, 1, 1, 10, 0, 0)
        assert time_diff(same_time, same_time) == 0

    def test_rounding(self) -> None:
        """Test rounding to nearest minute."""
        start = datetime.datetime(2024, 1, 1, 10, 0, 0)
        end = datetime.datetime(2024, 1, 1, 10, 5, 29)  # 5 min 29 sec
        assert time_diff(start, end) == 5
        end = datetime.datetime(2024, 1, 1, 10, 5, 31)  # 5 min 31 sec
        assert time_diff(start, end) == 6


class TestAlignTimestamps:
    """Tests for align_timestamps function."""

    def test_positive_offset(self) -> None:
        """Test subtracting hours from timestamps."""
        dates = pd.date_range("2024-01-01 12:00:00", periods=3, freq="h")
        df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)

        result = align_timestamps(df, offset_hours=2)

        expected_start = pd.Timestamp("2024-01-01 10:00:00")
        assert result.index[0] == expected_start

    def test_zero_offset(self) -> None:
        """Test zero offset (no change)."""
        dates = pd.date_range("2024-01-01 12:00:00", periods=3, freq="h")
        df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)
        original_index = df.index.copy()

        result = align_timestamps(df, offset_hours=0)

        assert all(result.index == original_index)

    def test_preserves_data(self) -> None:
        """Test that data values are preserved."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"value": [10, 20, 30]}, index=dates)

        result = align_timestamps(df, offset_hours=5)

        assert list(result["value"]) == [10, 20, 30]


class TestValidateCovarianceMatrix:
    """Tests for validate_covariance_matrix function."""

    def test_valid_matrix_unchanged(self) -> None:
        """Test that valid matrix passes through unchanged."""
        cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        result = validate_covariance_matrix(cov.copy())

        np.testing.assert_array_almost_equal(result.values, cov.values)

    def test_replaces_nan_diagonal(self) -> None:
        """Test NaN on diagonal replaced with 0.01."""
        cov = pd.DataFrame(
            [[np.nan, 0.01], [0.01, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        result = validate_covariance_matrix(cov)

        assert result.iloc[0, 0] == 0.01
        assert result.iloc[1, 1] == 0.09

    def test_replaces_nan_off_diagonal(self) -> None:
        """Test NaN off-diagonal replaced with 0."""
        cov = pd.DataFrame(
            [[0.04, np.nan], [np.nan, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        result = validate_covariance_matrix(cov)

        assert result.iloc[0, 1] == 0
        assert result.iloc[1, 0] == 0

    def test_replaces_inf(self) -> None:
        """Test Inf values are replaced appropriately."""
        cov = pd.DataFrame(
            [[np.inf, 0.01], [0.01, -np.inf]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        result = validate_covariance_matrix(cov)

        assert result.iloc[0, 0] == 0.01  # diagonal
        assert result.iloc[1, 1] == 0.01  # diagonal

    def test_preserves_index_columns(self) -> None:
        """Test that index and column names are preserved."""
        cov = pd.DataFrame(
            [[0.04, np.nan], [np.nan, 0.09]],
            index=["AAPL", "MSFT"],
            columns=["AAPL", "MSFT"],
        )
        result = validate_covariance_matrix(cov)

        assert list(result.index) == ["AAPL", "MSFT"]
        assert list(result.columns) == ["AAPL", "MSFT"]


class TestNearestPsd:
    """Tests for nearest_psd function."""

    def test_already_psd_matrix(self) -> None:
        """Test matrix that is already PSD passes through."""
        # Identity matrix is PSD
        psd = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        result = nearest_psd(psd)

        np.testing.assert_array_almost_equal(result.values, psd.values)

    def test_fixes_negative_eigenvalues(self) -> None:
        """Test that negative eigenvalues are clipped to positive."""
        # Matrix with negative eigenvalue
        non_psd = pd.DataFrame(
            [[1.0, 2.0], [2.0, 1.0]],  # eigenvalues are 3 and -1
            index=["A", "B"],
            columns=["A", "B"],
        )
        result = nearest_psd(non_psd)

        # Result should be PSD (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(result.values)
        assert all(e >= 0 for e in eigenvalues)

    def test_result_is_symmetric(self) -> None:
        """Test that result is symmetric."""
        non_psd = pd.DataFrame(
            [[1.0, 2.0], [2.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        result = nearest_psd(non_psd)

        np.testing.assert_array_almost_equal(result.values, result.values.T)

    def test_preserves_index_columns(self) -> None:
        """Test that index and column names are preserved."""
        cov = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=["AAPL", "MSFT"],
            columns=["AAPL", "MSFT"],
        )
        result = nearest_psd(cov)

        assert list(result.index) == ["AAPL", "MSFT"]
        assert list(result.columns) == ["AAPL", "MSFT"]

    def test_custom_tolerance(self) -> None:
        """Test custom tolerance parameter."""
        non_psd = pd.DataFrame(
            [[1.0, 2.0], [2.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        tol = 0.1
        result = nearest_psd(non_psd, tol=tol)

        eigenvalues = np.linalg.eigvalsh(result.values)
        assert all(e >= tol for e in eigenvalues)


class TestBuildExpWeightedCovarianceMatrix:
    """Tests for build_exp_weighted_covariance_matrix function."""

    @pytest.mark.parametrize("span", [2, 6, 12, 120])
    def test_final_correlation_matches_pandas_with_pairwise_gaps(self, span):
        rng = np.random.default_rng(431)
        values = rng.normal(0, 0.08, (90, 8))
        values[rng.random(values.shape) < 0.35] = np.nan
        values[:, 0] = 0.0
        values[-20:, 1] = 0.0
        values[:50, 2] = np.nan
        values[-10:, 3] = np.nan
        frame = pd.DataFrame(values)
        expected = (
            frame.ewm(span=span, min_periods=2, adjust=True, ignore_na=False)
            .corr(pairwise=True)
            .xs(frame.index[-1], level=0)
        )
        np.testing.assert_allclose(
            _final_ewm_correlation(frame, span),
            expected,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

    def test_covariance_does_not_materialize_correlation_history(self, monkeypatch):
        from pandas.core.window.ewm import ExponentialMovingWindow

        def forbidden(*args, **kwargs):
            raise AssertionError("Full-history correlation allocates T x N x N")

        monkeypatch.setattr(ExponentialMovingWindow, "corr", forbidden)
        frame = pd.DataFrame(np.random.default_rng(51).normal(0, 0.02, (40, 80)))
        covariance = build_exp_weighted_covariance_matrix(frame)
        assert np.isfinite(covariance).all().all()

    def test_unestimable_pair_names_are_reported(self):
        frame = pd.DataFrame(
            {"OLD": [0.1, -0.1, 0, 0, 0], "NEW": [np.nan, np.nan, 0.2, -0.1, 0.1]}
        )
        with pytest.raises(ValueError, match="OLD.*NEW"):
            build_exp_weighted_covariance_matrix(frame)

    def test_basic_covariance(self) -> None:
        """Test basic covariance matrix construction."""
        np.random.seed(42)
        returns = pd.DataFrame(
            {
                "A": np.random.randn(100) * 0.02,
                "B": np.random.randn(100) * 0.02,
            }
        )

        result = build_exp_weighted_covariance_matrix(returns)

        # Should be square with same assets
        assert result.shape == (2, 2)
        assert list(result.index) == ["A", "B"]
        assert list(result.columns) == ["A", "B"]

        # Diagonal should be positive (variances)
        assert result.iloc[0, 0] > 0
        assert result.iloc[1, 1] > 0

    def test_handles_nan_values(self) -> None:
        """Missing observations are excluded from each pair's estimate."""
        returns = pd.DataFrame(
            {
                "A": [0.01, np.nan, 0.02, -0.01, 0.03],
                "B": [0.02, 0.01, np.nan, -0.02, 0.01],
            }
        )

        result = build_exp_weighted_covariance_matrix(returns)

        # Should not have any NaN
        assert not result.isna().any().any()

    def test_handles_inf_values(self) -> None:
        """Test that Inf values are handled."""
        returns = pd.DataFrame(
            {
                "A": [0.01, np.inf, 0.02, -0.01, 0.03],
                "B": [0.02, 0.01, -np.inf, -0.02, 0.01],
            }
        )

        result = build_exp_weighted_covariance_matrix(returns)

        # Should not have any Inf
        assert not np.isinf(result.values).any()

    def test_clips_extreme_values(self) -> None:
        """Test that extreme return values are clipped."""
        returns = pd.DataFrame(
            {
                "A": [0.01, 0.5, 0.02, -0.5, 0.03],  # 0.5 and -0.5 should be clipped
                "B": [0.02, 0.01, 0.03, -0.02, 0.01],
            }
        )

        result = build_exp_weighted_covariance_matrix(returns)

        # Result should be valid
        assert result.shape == (2, 2)
        assert not result.isna().any().any()

    def test_symmetric_result(self) -> None:
        """Test that resulting matrix is symmetric."""
        np.random.seed(123)
        returns = pd.DataFrame(
            {
                "A": np.random.randn(50) * 0.02,
                "B": np.random.randn(50) * 0.02,
                "C": np.random.randn(50) * 0.02,
            }
        )

        result = build_exp_weighted_covariance_matrix(returns)

        # Check symmetry
        np.testing.assert_array_almost_equal(result.values, result.values.T, decimal=10)

    def test_single_asset(self) -> None:
        """Test with single asset."""
        returns = pd.DataFrame(
            {
                "A": [0.01, 0.02, -0.01, 0.03, -0.02],
            }
        )

        result = build_exp_weighted_covariance_matrix(returns)

        assert result.shape == (1, 1)
        assert result.iloc[0, 0] > 0  # Variance should be positive

    def test_short_series_fallback(self) -> None:
        """Two overlapping returns are regularized without a fallback model."""
        returns = pd.DataFrame(
            {
                "A": [0.01, 0.02],
                "B": [0.02, -0.01],
            }
        )

        result = build_exp_weighted_covariance_matrix(returns)

        # Should still produce valid 2x2 matrix
        assert result.shape == (2, 2)
        assert not result.isna().any().any()

    def test_newly_eligible_asset_preserves_older_asset_variance(self) -> None:
        """A 12-month IPO history must not truncate a mature asset's history."""
        rng = np.random.default_rng(104)
        mature = rng.normal(0.01, 0.05, 64)
        # Recent volatility differs deliberately from the older history.
        mature[-12:] *= 0.05
        frame = pd.DataFrame({"OLD": mature})
        original = build_exp_weighted_covariance_matrix(frame)
        frame["NEW"] = np.r_[np.full(52, np.nan), rng.normal(0.02, 0.08, 12)]
        combined = build_exp_weighted_covariance_matrix(frame)

        assert combined.loc["OLD", "OLD"] == pytest.approx(
            original.loc["OLD", "OLD"], rel=1e-12
        )
        assert not np.isclose(
            combined.loc["OLD", "OLD"],
            frame["OLD"].tail(12).ewm(span=6).var().iloc[-1],
            rtol=1e-3,
            atol=0,
        )
        counts = combined.attrs["covariance_diagnostics"]["observed_returns"]
        assert counts == {"OLD": 64, "NEW": 12}

    def test_missing_returns_are_not_repeated_or_backfilled(self) -> None:
        frame = pd.DataFrame(
            {
                "A": [0.05, -0.02, np.nan, 0.01, np.nan, -0.04, 0.02],
                "B": [np.nan, 0.03, -0.02, 0.01, 0.04, -0.03, 0.05],
            }
        )
        before = frame.copy()
        result = build_exp_weighted_covariance_matrix(frame)
        np.testing.assert_allclose(
            np.diag(result), frame.ewm(span=6, min_periods=2).var().iloc[-1]
        )
        assert result.attrs["covariance_diagnostics"]["minimum_pair_overlap"] == 4
        pd.testing.assert_frame_equal(frame, before)

    def test_short_wide_history_has_structural_risk_floor(self) -> None:
        """An optimizer cannot exploit rank deficiency to predict zero risk."""
        rng = np.random.default_rng(55)
        frame = pd.DataFrame(rng.normal(0, 0.05, (12, 80)))
        covariance = build_exp_weighted_covariance_matrix(frame)
        variances = np.diag(covariance)
        std = np.sqrt(variances)
        correlation = covariance.to_numpy() / np.outer(std, std)
        assert np.linalg.eigvalsh(correlation).min() >= 0.1 - 1e-12
        for _ in range(20):
            weights = rng.dirichlet(np.ones(80))
            independent_variance = float((weights**2) @ variances)
            assert float(weights @ covariance @ weights) >= (
                0.1 * independent_variance - 1e-14
            )

    def test_inconsistent_pairwise_correlations_are_repaired(self) -> None:
        # A/B and A/C positive, B/C negative, each observed on different dates.
        frame = pd.DataFrame(
            {
                "A": [-0.1, 0.1, -0.1, 0.1, np.nan, np.nan],
                "B": [-0.1, 0.1, np.nan, np.nan, -0.1, 0.1],
                "C": [np.nan, np.nan, -0.1, 0.1, 0.1, -0.1],
            }
        )
        covariance = build_exp_weighted_covariance_matrix(frame)
        diag = covariance.attrs["covariance_diagnostics"]
        assert diag["raw_correlation_min_eigenvalue"] < 0
        assert diag["regularized_correlation_min_eigenvalue"] >= 0.1 - 1e-12
        np.testing.assert_allclose(
            np.diag(covariance), frame.ewm(span=6, min_periods=2).var().iloc[-1]
        )

    def test_asset_order_and_return_units_are_preserved(self) -> None:
        rng = np.random.default_rng(6)
        frame = pd.DataFrame(rng.normal(0, 0.05, (40, 3)), columns=["Z", "A", "M"])
        original = build_exp_weighted_covariance_matrix(frame)
        scaled = build_exp_weighted_covariance_matrix(frame * 100)
        reordered = build_exp_weighted_covariance_matrix(frame[["M", "Z", "A"]])
        np.testing.assert_allclose(scaled, original * 10_000, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            reordered.loc[frame.columns, frame.columns], original
        )

    @pytest.mark.parametrize(
        "frame",
        [
            pd.DataFrame(),
            pd.DataFrame({"A": [0.01]}),
            pd.DataFrame({"A": [np.nan, np.nan]}),
            pd.DataFrame({"A": [0.0, 0.0, 0.0]}),
            pd.DataFrame(
                {"A": [0.01, 0.02, np.nan, np.nan], "B": [np.nan, np.nan, 0.02, 0.03]}
            ),
        ],
    )
    def test_unestimable_risk_raises_instead_of_inventing_covariance(
        self, frame: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError):
            build_exp_weighted_covariance_matrix(frame)

    @pytest.mark.parametrize("shrinkage", [0.0, -0.1, 1.01, np.nan, np.inf])
    def test_invalid_shrinkage(self, shrinkage: float) -> None:
        with pytest.raises(ValueError, match="shrinkage"):
            build_exp_weighted_covariance_matrix(
                pd.DataFrame({"A": [0.01, 0.02, -0.01]}), shrinkage=shrinkage
            )

    def test_full_shrinkage_retains_marginal_risk(self) -> None:
        frame = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.03, 0.04, -0.02]})
        result = build_exp_weighted_covariance_matrix(frame, shrinkage=1.0)
        expected = np.diag(frame.ewm(span=6).var().iloc[-1])
        np.testing.assert_allclose(result, expected)

    def test_chronological_sorting_and_duplicate_dates(self) -> None:
        frame = pd.DataFrame({"A": [0.01, 0.02, -0.01]}, index=[3, 1, 2])
        original = build_exp_weighted_covariance_matrix(frame)
        sorted_frame = build_exp_weighted_covariance_matrix(frame.sort_index())
        pd.testing.assert_frame_equal(original, sorted_frame)
        frame.index = [1, 1, 2]
        with pytest.raises(ValueError, match="unique"):
            build_exp_weighted_covariance_matrix(frame)
