"""
Output analysis module for the Investment Tool.

This package provides comprehensive output generation, analysis, and
reporting for both backtesting and live investment modes, including:

- **metrics**: Advanced performance metric computation
- **plots**: Professional interactive charts via Plotly
- **report**: Self-contained HTML summary reports
- **export**: Structured CSV/JSON data exports
- **output_manager**: Top-level façade orchestrating all output
"""

from investment_tool.output.export import DataExporter
from investment_tool.output.metrics import MetricsCalculator
from investment_tool.output.output_manager import OutputManager
from investment_tool.output.plots import PlotBuilder
from investment_tool.output.report import ReportBuilder
from investment_tool.output.strategy_comparator import StrategyComparator

__all__ = [
    "DataExporter",
    "MetricsCalculator",
    "OutputManager",
    "PlotBuilder",
    "ReportBuilder",
    "StrategyComparator",
]
