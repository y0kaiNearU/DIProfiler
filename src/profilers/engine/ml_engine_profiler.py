from __future__ import annotations

from typing import Any, Callable

from core.profiler import Profiler
from models.models import EngineRecommendation, EngineType, PipelineRequest, ProfilingResult
from profilers.common.features import extract
from profilers.common.ml_scoring import score_with_model


class MLEngineProfiler(Profiler):
    """
    Engine profiler backed by a scikit-learn classifier.

    The model must expose `predict_proba` and `classes_` (any sklearn classifier works).
    `classes_` values must match EngineType.value strings (e.g. "duckdb", "spark") by
    default; pass `candidate_type` to map classes_ values differently.

    Usage:
        profiler = MLEngineProfiler(my_fitted_model)
        result = profiler.profile(request)

    Custom feature extraction (e.g. a model trained on a different feature set):
        profiler = MLEngineProfiler(my_fitted_model, extract_fn=my_extract_fn)

    Persist / load:
        profiler.save("model.joblib")
        profiler = MLEngineProfiler.load("model.joblib")
    """

    def __init__(
        self,
        model: Any,
        extract_fn: Callable[[PipelineRequest], list[float]] = extract,
        candidate_type: Callable[[str], EngineType] = EngineType,
    ) -> None:
        self._model = model
        self._extract_fn = extract_fn
        self._candidate_type = candidate_type

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
        scored = score_with_model(self._model, self._extract_fn(request), self._candidate_type)

        recommendations = [
            EngineRecommendation(
                engine=engine,
                confidence=round(float(proba), 3),
                reasoning=f"ML model confidence {proba:.1%}",
            )
            for engine, proba in scored
            if engine in available and proba > 0
        ]
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return ProfilingResult(request=request, recommendations=recommendations)
