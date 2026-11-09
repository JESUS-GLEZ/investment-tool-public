"""Abstract base class for return prediction strategies.

All concrete prediction strategies must subclass
:class:`BasePredictionStrategy` and implement :meth:`predict_returns`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePredictionStrategy(ABC):
    """Interface for predicting next-period asset returns.

    Subclasses implement a single method that receives no arguments
    (all required data is injected at construction time) and returns a
    mapping from stock symbol to predicted return.
    """

    @abstractmethod
    def predict_returns(self) -> dict[str, float]:
        """Predict expected returns for the next period.

        Returns
        -------
        dict[str, float]
            Mapping from stock symbol to predicted expected return.
        """
