"""Tests for decisions.calibration module."""

import numpy as np
import pytest

from speculator_ev_engine.decisions.calibration import (
    full_calibration,
    calibration_correction,
    calibration_score,
)


class TestFullCalibration:
    def test_well_calibrated(self) -> None:
        rng = np.random.default_rng(42)
        # Generate forecasts that match their probabilities
        probs = rng.uniform(0.1, 0.9, 1000)
        outcomes = (rng.random(1000) < probs).astype(float)
        report = full_calibration(probs, outcomes)
        assert report.brier_score < 0.30  # Should be reasonable
        assert report.ece < 0.15  # Should be reasonably calibrated

    def test_poorly_calibrated(self) -> None:
        # All forecasts say 0.9 but outcomes are 50/50
        probs = np.full(100, 0.9)
        outcomes = np.random.default_rng(42).choice([0.0, 1.0], 100)
        report = full_calibration(probs, outcomes)
        assert report.brier_score > 0.1  # Should be poor

    def test_report_fields(self) -> None:
        probs = np.array([0.5, 0.5, 0.5])
        outcomes = np.array([1.0, 0.0, 1.0])
        report = full_calibration(probs, outcomes)
        assert report.brier_score >= 0.0
        assert 0.0 <= report.ece <= 1.0
        assert 0.0 <= report.mce <= 1.0


class TestCalibrationCorrection:
    def test_shape_preserved(self) -> None:
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.1, 0.9, 100)
        outcomes = (rng.random(100) < 0.5).astype(float)
        calibrated = calibration_correction(probs, outcomes)
        assert calibrated.shape == probs.shape

    def test_bounded_output(self) -> None:
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.1, 0.9, 100)
        outcomes = (rng.random(100) < 0.5).astype(float)
        calibrated = calibration_correction(probs, outcomes)
        assert np.all(calibrated >= 0.001)
        assert np.all(calibrated <= 0.999)


class TestCalibrationScore:
    def test_perfect_is_zero(self) -> None:
        score = calibration_score(0.0, 0.0)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_worse_is_higher(self) -> None:
        good = calibration_score(0.01, 0.1)
        bad = calibration_score(0.1, 0.3)
        assert bad > good
