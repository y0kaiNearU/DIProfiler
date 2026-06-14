from __future__ import annotations

from typing import Any

import numpy as np

from core.profiler import Profiler
from models.models import EngineRecommendation, EngineType, PipelineRequest, ProfilingResult
from profilers.features import extract


class MLEngineProfiler(Profiler):
    """
    Engine profiler backed by a scikit-learn classifier.

    The model must expose `predict_proba` and `classes_` (any sklearn classifier works).
    `classes_` values must match EngineType.value strings (e.g. "duckdb", "spark").

    Usage:
        profiler = MLEngineProfiler(my_fitted_model)
        result = profiler.profile(request)

    Persist / load:
        profiler.save("model.joblib")
        profiler = MLEngineProfiler.load("model.joblib")
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    # ------------------------------------------------------------------ #
    # Factory helpers                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str) -> MLEngineProfiler:
        import joblib
        return cls(joblib.load(path))

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self._model, path)

    # ------------------------------------------------------------------ #
    # Profiler interface                                                   #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "ml_engine_profiler"

    def can_handle(self, request: PipelineRequest) -> bool:
        return True

    def profile(self, request: PipelineRequest) -> ProfilingResult:
        available = set(request.available_engines)
        features = np.array([extract(request)], dtype=np.float32)

        classes = [EngineType(c) for c in self._model.classes_]
        probas = self._model.predict_proba(features)[0]

        recommendations = [
            EngineRecommendation(
                engine=engine,
                confidence=round(float(proba), 3),
                reasoning=f"ML model confidence {proba:.1%}",
            )
            for engine, proba in zip(classes, probas)
            if engine in available and proba > 0
        ]
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return ProfilingResult(request=request, recommendations=recommendations)
