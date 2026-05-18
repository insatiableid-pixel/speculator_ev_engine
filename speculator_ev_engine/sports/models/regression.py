"""Logistic regression baseline for sports prediction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.linear_model import LogisticRegression as SklearnLR

from .base_model import BaseModel, ModelMetrics
from ...core.distributions import brier_score, calibration_curve


# TODO: Add L1/L2 regularization tuning
# TODO: Add feature importance extraction
# TODO: Add calibration-aware training (temperature scaling)


class LogisticRegressionModel(BaseModel):
    """Logistic regression baseline for binary outcome prediction.

    Wraps sklearn's LogisticRegression with the project's BaseModel interface
    and adds calibration-aware evaluation.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self._model = SklearnLR(C=C, max_iter=max_iter, solver="lbfgs")
        self._is_fitted = False

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit logistic regression model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary target array (n_samples,).
        """
        self._model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return predicted probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of probabilities in [0, 1].
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self._model.predict_proba(X)[:, 1]

    def evaluate(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> ModelMetrics:
        """Evaluate model on held-out data.

        Args:
            X: Feature matrix.
            y: True targets.

        Returns:
            ModelMetrics.
        """
        probs = self.predict(X)
        y_arr = np.asarray(y, dtype=np.float64)

        # Log loss
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        log_loss = float(-np.mean(y_arr * np.log(probs_clipped) + (1 - y_arr) * np.log(1 - probs_clipped)))

        bs = brier_score(probs, y_arr)
        accuracy = float(np.mean((probs > 0.5) == y_arr))

        # ECE
        centers, freqs, counts = calibration_curve(probs, y_arr, n_bins=10)
        ece = float(np.sum(counts * np.abs(centers - freqs)) / max(np.sum(counts), 1))

        return ModelMetrics(
            log_loss=log_loss,
            brier_score=bs,
            accuracy=accuracy,
            calibration_error=ece,
            n_samples=len(y_arr),
        )
