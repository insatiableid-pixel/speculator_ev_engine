"""Ensemble framework for combining predictive models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .base_model import BaseModel, ModelMetrics
from ...core.distributions import brier_score


# TODO: Implement stacking ensemble
# TODO: Implement weighted averaging with learned weights
# TODO: Implement diversity-aware ensemble weighting
# TODO: Implement ensemble pruning


@dataclass
class EnsembleModel(BaseModel):
    """Weighted ensemble of multiple predictive models.

    Combines predictions from multiple models using configurable weights.
    Weights are normalized to sum to 1.

    Attributes:
        models: List of fitted BaseModel instances.
        weights: Weight per model (normalized internally).
    """
    models: list[BaseModel] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.models) != len(self.weights):
            raise ValueError("models and weights must have same length")
        if len(self.weights) > 0:
            total = sum(self.weights)
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            self.weights = [w / total for w in self.weights]

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit all constituent models.

        Args:
            X: Feature matrix.
            y: Target array.
        """
        for model in self.models:
            model.fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return weighted average of model predictions.

        Args:
            X: Feature matrix.

        Returns:
            Weighted probability predictions.
        """
        if not self.models:
            raise RuntimeError("Ensemble has no models")
        predictions = np.array([m.predict(X) for m in self.models])
        weights = np.array(self.weights).reshape(-1, 1)
        return np.sum(predictions * weights, axis=0)

    def evaluate(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> ModelMetrics:
        """Evaluate ensemble on held-out data.

        Args:
            X: Feature matrix.
            y: True targets.

        Returns:
            ModelMetrics.
        """
        probs = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64)
        bs = brier_score(probs, y_arr)
        accuracy = float(np.mean((probs > 0.5) == y_arr))

        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        log_loss = float(-np.mean(y_arr * np.log(probs_clipped) + (1 - y_arr) * np.log(1 - probs_clipped)))

        return ModelMetrics(
            log_loss=log_loss,
            brier_score=bs,
            accuracy=accuracy,
            calibration_error=0.0,  # TODO: implement
            n_samples=len(y_arr),
        )

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble.

        Args:
            model: Fitted BaseModel.
            weight: Relative weight for this model.
        """
        self.models.append(model)
        self.weights.append(weight)
        # Re-normalize
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
