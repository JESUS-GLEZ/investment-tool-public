"""
Self-contained HTML report generation for investment analysis.

This module provides the ReportBuilder class which generates a comprehensive
single-file HTML summary report embedding all key metrics, tables, and
inline chart images.
"""

from __future__ import annotations
from numbers import Real

import base64
import logging
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Generate a self-contained HTML performance report.

    Creates a single HTML file with embedded CSS that summarises all
    key metrics, return breakdowns, and optionally embeds plot images
    as base64 data URIs.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics dictionary from ``MetricsCalculator.compute_all()``.
    output_dir : Path
        Directory where the report will be saved.
    title : str, optional
        Report title, by default "Investment Performance Report".
    rebalancing_detail : pd.DataFrame | None, optional
        Per-asset weights and prices at each rebalancing period.
    """

    def __init__(
        self,
        metrics: dict[str, Any],
        output_dir: Path,
        title: str = "Investment Performance Report",
        rebalancing_detail: pd.DataFrame | None = None,
    ) -> None:
        self._metrics = metrics
        self._output_dir = output_dir
        self._title = title
        self._rebalancing_detail = rebalancing_detail

    def build(self) -> str:
        """Generate the HTML report and save to disk.

        Returns
        -------
        str
            File path of the saved report.
        """
        html = self._render()
        path = self._output_dir / "report.html"
        path.write_text(html, encoding="utf-8")
        logger.info(f"Saved HTML report: {path}")
        return str(path)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> str:
        """Render the full HTML document.

        Returns
        -------
        str
            Complete HTML string.
        """
        m = self._metrics
        sections = [
            self._section_summary(m),
            self._section_risk(m),
            self._section_win_loss(m),
            self._section_rebalancing_detail(self._rebalancing_detail),
            self._section_monthly_returns(m),
            self._section_yearly_returns(m),
            self._section_portfolio(m),
            self._section_embedded_plots(),
        ]
        body = "\n".join(sections)

        return _HTML_TEMPLATE.format(
            title=self._title,
            body=body,
        )

    # ------------------------------------------------------------------
    # Report sections
    # ------------------------------------------------------------------

    @staticmethod
    def _section_summary(m: dict[str, Any]) -> str:
        """Render the summary / key metrics section.

        Parameters
        ----------
        m : dict[str, Any]
            Metrics dictionary.

        Returns
        -------
        str
            HTML fragment.
        """
        rows = [
            ("Total Return", _fmt_pct(m.get("total_return"))),
            (
                _annual_return_label(m.get("metric_definitions")),
                _fmt_pct(m.get("annualized_return")),
            ),
            ("Annualised Volatility", _fmt_pct(m.get("annualized_volatility"))),
            ("Sharpe Ratio", _fmt_num(m.get("sharpe_ratio"))),
            ("Sortino Ratio", _fmt_num(m.get("sortino_ratio"))),
            ("Calmar Ratio", _fmt_num(m.get("calmar_ratio"))),
            ("Max Drawdown", _fmt_pct(m.get("max_drawdown"))),
            ("Total Periods", _fmt_count(m.get("total_periods"))),
            ("Initial Value", _fmt_money(m.get("initial_value"))),
            ("Final Value", _fmt_money(m.get("final_value"))),
        ]
        return _card("Summary", _table(rows))

    @staticmethod
    def _section_risk(m: dict[str, Any]) -> str:
        """Render the risk metrics section.

        Parameters
        ----------
        m : dict[str, Any]
            Metrics dictionary.

        Returns
        -------
        str
            HTML fragment.
        """
        rows = [
            (
                "Max Drawdown Duration (periods)",
                _fmt_count(m.get("max_drawdown_duration_periods")),
            ),
            ("Drawdown Start", str(m.get("max_drawdown_start", "—"))),
            ("Drawdown End", str(m.get("max_drawdown_end", "—"))),
            ("Average Return", _fmt_pct(m.get("average_return"))),
            ("Std Return", _fmt_pct(m.get("std_return"))),
        ]
        return _card("Risk Analysis", _table(rows))

    @staticmethod
    def _section_win_loss(m: dict[str, Any]) -> str:
        """Render the win/loss section.

        Parameters
        ----------
        m : dict[str, Any]
            Metrics dictionary.

        Returns
        -------
        str
            HTML fragment.
        """
        rows = [
            ("Win Rate", _fmt_pct(m.get("win_rate"))),
            ("Profit Factor", _fmt_num(m.get("profit_factor"))),
            (
                "Best Period",
                f"{_fmt_pct(m.get('best_period'))}  ({m.get('best_period_date', '—')})",
            ),
            (
                "Worst Period",
                f"{_fmt_pct(m.get('worst_period'))}  ({m.get('worst_period_date', '—')})",
            ),
        ]
        return _card("Win / Loss", _table(rows))

    @staticmethod
    def _section_rebalancing_detail(detail: pd.DataFrame | None) -> str:
        """Render the rebalancing detail table.

        Shows selected assets, their target weights, period closing prices,
        next-period closing prices and the resulting price change percentage.

        Parameters
        ----------
        detail : pd.DataFrame | None
            Rebalancing detail with columns: date, symbol, weight, price,
            next_price, price_change_pct.

        Returns
        -------
        str
            HTML fragment.
        """
        if detail is None or detail.empty:
            return ""

        has_period_dates = "period_end" in detail and detail["period_end"].notna().any()
        date_heading = "Period ending" if has_period_dates else "Date"
        execution_heading = "<th>Execution time (UTC)</th>" if has_period_dates else ""
        price_headings = (
            "<th>Close price</th><th>Next close</th>"
            if has_period_dates
            else "<th>Price</th><th>Next Price</th>"
        )
        header = (
            "<tr>"
            f"<th>{date_heading}</th>{execution_heading}<th>Symbol</th><th>Weight</th>"
            f"{price_headings}<th>Change %</th>"
            "</tr>"
        )

        body_rows: list[str] = []
        prev_date: str | None = None

        for _, row in detail.iterrows():
            dt_str = str(pd.Timestamp(row["date"]).date())
            execution_cell = ""
            if has_period_dates:
                execution = row.get("execution_time")
                execution_cell = f"<td>{escape(str(execution)) if pd.notna(execution) else 'Unavailable'}</td>"
            # Add visual separator between periods
            separator = ""
            if prev_date is not None and dt_str != prev_date:
                separator = ' style="border-top:2px solid #1f77b4"'
            prev_date = dt_str

            weight_str = f"{row['weight']:.2%}" if pd.notna(row["weight"]) else "—"
            price_str = f"${row['price']:.2f}" if pd.notna(row["price"]) else "—"
            next_str = (
                f"${row['next_price']:.2f}" if pd.notna(row["next_price"]) else "—"
            )

            change = row.get("price_change_pct")
            if pd.notna(change):
                colour = "#2ca02c" if change >= 0 else "#d62728"
                change_str = f'<span style="color:{colour}">{change:+.2f}%</span>'
            else:
                change_str = "—"

            body_rows.append(
                f"<tr{separator}>"
                f"<td>{dt_str}</td>{execution_cell}<td>{row['symbol']}</td>"
                f"<td>{weight_str}</td><td>{price_str}</td>"
                f"<td>{next_str}</td><td>{change_str}</td>"
                f"</tr>"
            )

        table = f"<table>{header}{''.join(body_rows)}</table>"
        return _card("Rebalancing Detail", table)

    @staticmethod
    def _section_monthly_returns(m: dict[str, Any]) -> str:
        """Render the monthly returns heatmap table.

        Parameters
        ----------
        m : dict[str, Any]
            Metrics dictionary.

        Returns
        -------
        str
            HTML fragment.
        """
        monthly: pd.DataFrame | None = m.get("monthly_returns")
        if monthly is None or (isinstance(monthly, pd.DataFrame) and monthly.empty):
            return ""

        header = (
            "<tr><th>Year</th>"
            + "".join(f"<th>{c}</th>" for c in monthly.columns)
            + "</tr>"
        )
        body_rows: list[str] = []
        for year, row in monthly.iterrows():
            cells = f"<td><strong>{year}</strong></td>"
            for val in row:
                if pd.isna(val):
                    cells += "<td>—</td>"
                else:
                    colour = "#2ca02c" if val >= 0 else "#d62728"
                    cells += f'<td style="color:{colour}">{val * 100:.1f}%</td>'
            body_rows.append(f"<tr>{cells}</tr>")

        table = f"<table>{header}{''.join(body_rows)}</table>"
        return _card("Monthly Returns", table)

    @staticmethod
    def _section_yearly_returns(m: dict[str, Any]) -> str:
        """Render the yearly returns table.

        Parameters
        ----------
        m : dict[str, Any]
            Metrics dictionary.

        Returns
        -------
        str
            HTML fragment.
        """
        yearly: pd.Series | None = m.get("yearly_returns")
        if yearly is None or (isinstance(yearly, pd.Series) and yearly.empty):
            return ""

        rows = [(str(y), _fmt_pct(v)) for y, v in yearly.items()]
        return _card("Yearly Returns", _table(rows))

    @staticmethod
    def _section_portfolio(m: dict[str, Any]) -> str:
        """Render the portfolio / concentration section.

        Parameters
        ----------
        m : dict[str, Any]
            Metrics dictionary.

        Returns
        -------
        str
            HTML fragment.
        """
        rows = [
            ("Average Gross Turnover", _fmt_pct(m.get("average_turnover"))),
            ("Total Gross Turnover", _fmt_pct(m.get("total_turnover"))),
            ("Total Transaction Costs", _fmt_money(m.get("total_transaction_costs"))),
            ("Final Capital HHI (0–1)", _fmt_num(m.get("final_hhi"))),
            ("Final Top-3 Weight", _fmt_pct(m.get("final_top3_weight"))),
        ]
        definitions = "<p>Capital HHI sums squared risky-asset weights relative to total portfolio value; cash is excluded. Gross turnover counts buys plus sells relative to post-rebalance value, including recorded zero-trade periods and initial allocation. It is not halved or annualized; its sum can exceed 100%.</p>"
        return _card("Portfolio & Concentration", _table(rows) + definitions)

    def _section_embedded_plots(self) -> str:
        """Embed any PNG plots found in the plots directory as base64.

        Returns
        -------
        str
            HTML fragment with inline images.
        """
        plots_dir = self._output_dir / "plots"
        if not plots_dir.exists():
            return ""

        images: list[str] = []
        for png in sorted(plots_dir.glob("*.png")):
            try:
                data = base64.b64encode(png.read_bytes()).decode("ascii")
                images.append(
                    f'<div class="plot"><h3>{png.stem.replace("_", " ").title()}</h3>'
                    f'<img src="data:image/png;base64,{data}" alt="{png.stem}"></div>'
                )
            except Exception:
                logger.debug(f"Could not embed plot: {png}")

        if not images:
            return ""

        return _card("Charts", "\n".join(images))


# ======================================================================
# Helpers
# ======================================================================


def _finite_number(val: Any) -> float | None:
    if isinstance(val, bool) or not isinstance(val, Real):
        return None
    number = float(val)
    return number if np.isfinite(number) else None


def _annual_return_label(definitions: Any) -> str:
    if isinstance(definitions, dict) and definitions.get("version") == 2:
        if definitions.get("annualized_return") == "cagr":
            return "CAGR"
        if definitions.get("annualized_return") == "geometric_period_return":
            return "Annualized Compound Return"
    return "Annualized Return (definition unavailable)"


def _fmt_count(val: Any) -> str:
    number = _finite_number(val)
    return f"{number:,.0f}" if number is not None and number.is_integer() else "—"


def _fmt_pct(val: Any) -> str:
    """Format a value as a percentage string.

    Parameters
    ----------
    val : Any
        Numeric value or None.

    Returns
    -------
    str
        Formatted percentage or '—'.
    """
    number = _finite_number(val)
    if number is None:
        return "—"
    return f"{number * 100:.2f}%"


def _fmt_num(val: Any) -> str:
    """Format a value as a decimal string.

    Parameters
    ----------
    val : Any
        Numeric value or None.

    Returns
    -------
    str
        Formatted number or '—'.
    """
    number = _finite_number(val)
    if number is None:
        return "—"
    return f"{number:.2f}"


def _fmt_money(val: Any) -> str:
    """Format a value as a dollar amount.

    Parameters
    ----------
    val : Any
        Numeric value or None.

    Returns
    -------
    str
        Formatted dollar string or '—'.
    """
    number = _finite_number(val)
    if number is None:
        return "—"
    return f"${number:,.2f}"


def _table(rows: list[tuple[str, str]]) -> str:
    """Build a simple two-column HTML table.

    Parameters
    ----------
    rows : list[tuple[str, str]]
        (label, value) pairs.

    Returns
    -------
    str
        HTML table string.
    """
    body = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows
    )
    return f"<table>{body}</table>"


def _card(title: str, content: str) -> str:
    """Wrap content in a styled card div.

    Parameters
    ----------
    title : str
        Card heading.
    content : str
        Inner HTML content.

    Returns
    -------
    str
        HTML card fragment.
    """
    return f'<div class="card"><h2>{title}</h2>{content}</div>'


# ── HTML template ───────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{ --primary: #1f77b4; --bg: #f8f9fa; --card: #ffffff; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
          sans-serif; background: var(--bg); color: #333; padding: 2rem; }}
  h1 {{ text-align: center; color: var(--primary); margin-bottom: 2rem; }}
  .card {{ background: var(--card); border-radius: 8px; padding: 1.5rem;
           margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .card h2 {{ color: var(--primary); border-bottom: 2px solid var(--primary);
              padding-bottom: .4rem; margin-bottom: 1rem; font-size: 1.2rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  td, th {{ padding: .4rem .8rem; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f0f4f8; }}
  tr:hover {{ background: #f5f7fa; }}
  .plot img {{ max-width: 100%; height: auto; margin-top: .5rem; }}
  .plot h3 {{ font-size: 1rem; color: #555; }}
</style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>"""
