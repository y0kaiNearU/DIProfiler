import io
import tempfile

import numpy as np
import pytest

from models.models import (
    DatasetInfo,
    EngineType,
    FileFormat,
    FileSource,
    PipelineRequest,
)
from profilers.ml_engine_profiler import MLEngineProfiler


class _MockModel:
    """Minimal sklearn-compatible classifier."""

    classes_ = ["duckdb", "spark", "datafusion"]

    def predict_proba(self, X):
        # Return fixed probabilities regardless of input
        return np.array([[0.7, 0.2, 0.1]] * len(X))


class _HighSparkModel:
    classes_ = ["duckdb", "spark", "datafusion"]

    def predict_proba(self, X):
        return np.array([[0.1, 0.8, 0.1]] * len(X))


def _req(size_bytes=500 * 1024 ** 2, available_engines=None):
    return PipelineRequest(
        source=DatasetInfo(
            source=FileSource(path="x.csv", format=FileFormat.CSV),
            size_bytes=size_bytes,
        ),
        available_engines=available_engines if available_engines is not None else list(EngineType),
    )


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_name():
    assert MLEngineProfiler(_MockModel()).name == "ml_engine_profiler"


def test_can_handle_always_true():
    profiler = MLEngineProfiler(_MockModel())
    assert profiler.can_handle(_req())


def test_profile_returns_sorted_recommendations():
    profiler = MLEngineProfiler(_MockModel())
    result = profiler.profile(_req())
    confidences = [r.confidence for r in result.recommendations]
    assert confidences == sorted(confidences, reverse=True)


def test_profile_best_is_highest_confidence():
    profiler = MLEngineProfiler(_MockModel())
    result = profiler.profile(_req())
    assert result.best == result.recommendations[0]


def test_profile_uses_model_probabilities():
    profiler = MLEngineProfiler(_HighSparkModel())
    result = profiler.profile(_req())
    assert result.best.engine == EngineType.SPARK


def test_excludes_engines_not_in_available():
    profiler = MLEngineProfiler(_MockModel())
    req = _req(available_engines=[EngineType.DUCKDB, EngineType.SPARK])
    result = profiler.profile(req)
    engines = {r.engine for r in result.recommendations}
    assert EngineType.DATAFUSION not in engines


def test_only_one_engine_available():
    profiler = MLEngineProfiler(_MockModel())
    req = _req(available_engines=[EngineType.SPARK])
    result = profiler.profile(req)
    assert len(result.recommendations) == 1
    assert result.recommendations[0].engine == EngineType.SPARK


def test_zero_probability_engines_excluded():
    class ZeroSparkModel:
        classes_ = ["duckdb", "spark", "datafusion"]

        def predict_proba(self, X):
            return np.array([[0.9, 0.0, 0.1]] * len(X))

    profiler = MLEngineProfiler(ZeroSparkModel())
    result = profiler.profile(_req())
    engines = {r.engine for r in result.recommendations}
    assert EngineType.SPARK not in engines


def test_confidence_is_rounded_to_three_decimal_places():
    class PreciseModel:
        classes_ = ["duckdb", "spark", "datafusion"]

        def predict_proba(self, X):
            return np.array([[0.123456789, 0.5, 0.376543211]] * len(X))

    profiler = MLEngineProfiler(PreciseModel())
    result = profiler.profile(_req())
    for rec in result.recommendations:
        assert rec.confidence == round(rec.confidence, 3)


def test_save_and_load_round_trip(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    y = np.array(["duckdb", "spark"])
    model = RandomForestClassifier(n_estimators=2, random_state=0)
    model.fit(X, y)

    profiler = MLEngineProfiler(model)
    path = str(tmp_path / "model.joblib")
    profiler.save(path)

    loaded = MLEngineProfiler.load(path)
    assert list(loaded._model.classes_) == list(model.classes_)
