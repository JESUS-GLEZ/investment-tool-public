"""Unit tests for PredictorConfig dataclass."""

from __future__ import annotations

from investment_tool.strategies.prediction_strategies.average_prediction_strategy import (
    AveragePredictionStrategy,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison.predictor_config import (
    PredictorConfig,
)


class TestPredictorConfig:
    def test_basic_construction(self) -> None:
        cfg = PredictorConfig(
            label="EWM-6",
            factory=lambda s, ss: AveragePredictionStrategy(s, span=6),
        )
        assert cfg.label == "EWM-6"
        assert callable(cfg.factory)

    def test_frozen(self) -> None:
        cfg = PredictorConfig(
            label="Test",
            factory=lambda s, ss: AveragePredictionStrategy(s),
        )
        try:
            cfg.label = "changed"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass
