"""Predictor configuration dataclass.

Describes a single prediction strategy to benchmark: the factory
callable that builds it from (stocks, sentiment_scores) and any
keyword arguments.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.prediction_strategies.base_prediction_strategy import (
    BasePredictionStrategy,
)

# The factory receives (stocks, sentiment_scores, **kwargs) and returns
# a configured BasePredictionStrategy.
PredictorFactory = Callable[
    [Sequence[Stock], dict[str, dict[date, float]]],
    BasePredictionStrategy,
]


@dataclass(frozen=True)
class PredictorConfig:
    """Immutable description of one prediction strategy to benchmark.

    Parameters
    ----------
    label : str
        Human-readable name (e.g. ``"RidgeVARX"``, ``"EWM-6"``).
    factory : PredictorFactory
        Callable ``(stocks, sentiment_scores) -> BasePredictionStrategy``.
        Use ``functools.partial`` or a lambda to bake in extra kwargs
        (e.g. ``lambda s, ss: AveragePredictionStrategy(s, span=12)``).
    """

    label: str
    factory: PredictorFactory
