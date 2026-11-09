"""Portfolio optimisation models for Platypus evolutionary algorithms."""

from __future__ import annotations

from typing import Union

from .bprc import BalancedPortfoliosRiskContributions
from .markowitz import MarkowitzLongOnly

PortfolioModel = Union[MarkowitzLongOnly, BalancedPortfoliosRiskContributions]

__all__ = [
    "BalancedPortfoliosRiskContributions",
    "MarkowitzLongOnly",
    "PortfolioModel",
]
