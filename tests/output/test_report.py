"""Tests for ReportBuilder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from investment_tool.output.report import ReportBuilder


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_metrics() -> dict[str, Any]:
    return {
        "total_return": 0.12,
        "annualized_return": 0.15,
        "annualized_volatility": 0.10,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "calmar_ratio": 3.0,
        "max_drawdown": -0.05,
        "max_drawdown_duration_periods": 3,
        "max_drawdown_start": "2024-03-15",
        "max_drawdown_end": "2024-06-15",
        "average_return": 0.01,
        "std_return": 0.02,
        "win_rate": 0.75,
        "profit_factor": 2.5,
        "best_period": 0.05,
        "worst_period": -0.03,
        "best_period_date": "2024-06-15",
        "worst_period_date": "2024-02-15",
        "average_turnover": 0.15,
        "total_turnover": 1.8,
        "final_hhi": 0.35,
        "final_top3_weight": 0.90,
        "initial_value": 10000.0,
        "final_value": 11200.0,
        "total_periods": 12,
        "monthly_returns": pd.DataFrame(
            [[0.01, 0.02, -0.01]],
            index=[2024],
            columns=["Jan", "Feb", "Mar"],
        ),
        "yearly_returns": pd.Series([0.12], index=[2024]),
    }


# ── Tests ───────────────────────────────────────────────────────────────────


class TestReportBuilder:
    """Test HTML report generation."""

    @pytest.mark.parametrize(
        "definitions,label",
        [
            ({"version": 2, "annualized_return": "cagr"}, "CAGR"),
            (
                {"version": 2, "annualized_return": "geometric_period_return"},
                "Annualized Compound Return",
            ),
            (None, "Annualized Return (definition unavailable)"),
            ("legacy", "Annualized Return (definition unavailable)"),
            (
                {"annualized_return": "cagr"},
                "Annualized Return (definition unavailable)",
            ),
            (
                {"version": 2, "annualized_return": "unknown"},
                "Annualized Return (definition unavailable)",
            ),
        ],
    )
    def test_return_label_requires_recognized_metadata(
        self, definitions, label, tmp_path
    ):
        metrics = {"annualized_return": 0.12, "metric_definitions": definitions}
        report = Path(ReportBuilder(metrics, tmp_path).build()).read_text()
        assert label in report
        assert "12.00%" in report

    def test_ratio_units_scale_and_turnover_match_frontend_contract(self, tmp_path):
        metrics = {
            "profit_factor": 1.5,
            "final_hhi": 0.25,
            "average_turnover": 0.15,
            "total_turnover": 1.8,
        }
        report = Path(ReportBuilder(metrics, tmp_path).build()).read_text()
        for expected in [
            "1.50",
            "0.25",
            "15.00%",
            "180.00%",
            "Final Capital HHI (0–1)",
            "Average Gross Turnover",
            "Total Gross Turnover",
            "not halved or annualized",
        ]:
            assert expected in report
        assert "1.5000" not in report
        assert "150.00%" not in report

    @pytest.mark.parametrize(
        "value", [None, float("nan"), float("inf"), float("-inf"), "Infinity", True, {}]
    )
    def test_unavailable_numeric_values_do_not_escape_into_report(
        self, value, tmp_path
    ):
        metrics = {
            key: value
            for key in [
                "annualized_return",
                "profit_factor",
                "initial_value",
                "total_periods",
                "max_drawdown_duration_periods",
            ]
        }
        report = Path(ReportBuilder(metrics, tmp_path).build()).read_text()
        assert "—" in report
        for invalid in ["nan", "inf%", "$inf", "∞", "Infinity", "None", "True"]:
            assert f">{invalid}<" not in report

    def test_generates_html_file(
        self, sample_metrics: dict[str, Any], tmp_path: Path
    ) -> None:
        builder = ReportBuilder(sample_metrics, tmp_path)
        path = builder.build()
        assert Path(path).exists()
        assert path.endswith(".html")

    def test_html_contains_title(
        self, sample_metrics: dict[str, Any], tmp_path: Path
    ) -> None:
        builder = ReportBuilder(sample_metrics, tmp_path)
        path = builder.build()
        content = Path(path).read_text()
        assert "Investment Performance Report" in content

    def test_custom_title(self, sample_metrics: dict[str, Any], tmp_path: Path) -> None:
        builder = ReportBuilder(sample_metrics, tmp_path, title="Custom Report")
        path = builder.build()
        content = Path(path).read_text()
        assert "Custom Report" in content

    def test_contains_key_metrics(
        self, sample_metrics: dict[str, Any], tmp_path: Path
    ) -> None:
        builder = ReportBuilder(sample_metrics, tmp_path)
        path = builder.build()
        content = Path(path).read_text()
        assert "Sharpe Ratio" in content
        assert "Max Drawdown" in content
        assert "Win Rate" in content

    def test_monthly_returns_table(
        self, sample_metrics: dict[str, Any], tmp_path: Path
    ) -> None:
        builder = ReportBuilder(sample_metrics, tmp_path)
        path = builder.build()
        content = Path(path).read_text()
        assert "Monthly Returns" in content
        assert "Jan" in content

    def test_empty_metrics_no_crash(self, tmp_path: Path) -> None:
        builder = ReportBuilder({}, tmp_path)
        path = builder.build()
        assert Path(path).exists()

    def test_embeds_png_plots(
        self, sample_metrics: dict[str, Any], tmp_path: Path
    ) -> None:
        # Create a fake PNG in plots/
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()
        fake_png = plots_dir / "test_chart.png"
        # Minimal valid PNG (1x1 white pixel)
        fake_png.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        builder = ReportBuilder(sample_metrics, tmp_path)
        path = builder.build()
        content = Path(path).read_text()
        assert "data:image/png;base64," in content

    def test_rebalancing_detail_section_rendered(
        self, sample_metrics: dict[str, Any], tmp_path: Path
    ) -> None:
        detail = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-15")],
                "symbol": ["AAPL", "GOOG"],
                "weight": [0.6, 0.4],
                "price": [150.0, 200.0],
                "next_price": [155.0, 210.0],
                "price_change_pct": [3.33, 5.0],
            }
        )
        builder = ReportBuilder(sample_metrics, tmp_path, rebalancing_detail=detail)
        path = builder.build()
        content = Path(path).read_text()
        assert "Rebalancing Detail" in content
        assert "AAPL" in content
        assert "GOOG" in content
        assert "$150.00" in content
        assert "+3.33%" in content

    def test_no_rebalancing_section_when_empty(
        self, sample_metrics: dict[str, Any], tmp_path: Path
    ) -> None:
        builder = ReportBuilder(sample_metrics, tmp_path, rebalancing_detail=None)
        path = builder.build()
        content = Path(path).read_text()
        assert "Rebalancing Detail" not in content
