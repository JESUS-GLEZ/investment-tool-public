"""Prediction strategy comparison framework.

Provides tools to benchmark multiple return prediction strategies
via walk-forward evaluation, compute accuracy metrics, and generate
comparison reports.
"""

from .prediction_benchmark import PredictionBenchmark
from .predictor_config import PredictorConfig
from .predictor_result import PredictorResult
