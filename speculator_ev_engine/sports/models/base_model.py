"""Abstract base for any predictive model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# TODO: Implement full model lifecycle: train, predict, evaluate, serialize


@dataclass(frozen=True)
class ModelMetrics:
    """Standard evaluation metrics for a predictive model.

    Attributes:
        log_loss: Logarithmic loss.
        brier_score: Brier score.
        accuracy: Classification accuracy.
        calibration_error: Expected calibration error.
        n_samples: Number of samples used for evaluation.
    """
    log_loss: float
    brier_score: float
    accuracy: float
    calibration_error: float
    n_samples: int


class BaseModel(ABC):
    """Abstract base class for all predictive models.

    Every model must implement fit, predict, and evaluate.
    """

    @abstractmethod
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Train the model on features X and targets y.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target array (n_samples,).
        """
        ...

    @abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return predicted probabilities for each sample.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of predicted probabilities in [0, 1].
        """
        ...

    @abstractmethod
    def evaluate(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> ModelMetrics:
        """Evaluate model performance on held-out data.

        Args:
            X: Feature matrix.
            y: True targets.

        Returns:
            ModelMetrics with standard evaluation scores.
        """
        ...
