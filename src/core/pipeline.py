from __future__ import annotations

import dataclasses
import importlib.util

import narwhals as nw

from core.loader import Loader
from core.profiler import Profiler
from core.registry import LoaderRegistry, ProfilerRegistry, WriterRegistry
from core.writer import Writer
from engines.duckdb.loader import DuckDBLoader
from engines.duckdb.writer import DuckDBWriter
from engines.spark.loader import SparkLoader
from engines.spark.writer import SparkWriter
from models.models import EngineType, PipelineRequest, ProfilingResult
from profilers.engine_profiler import RuleBasedEngineProfiler

_ENGINE_MODULES: dict[EngineType, str] = {
    EngineType.DUCKDB: "duckdb",
    EngineType.SPARK: "pyspark",
}


def _detect_available_engines() -> list[EngineType]:
    return [e for e, mod in _ENGINE_MODULES.items() if importlib.util.find_spec(mod) is not None]


class DIProfiler:
    """
    Profiles datasets and provides engine recommendations, loading, and writing utilities.

    Usage:
        profiler = DIProfiler()

        # Profile and get engine recommendation
        result = profiler.recommend(request)

        # Load into a narwhals LazyFrame
        frame = profiler.load(request)

        # Apply custom transforms (not provided by DIProfiler)
        frame = my_transforms(frame)

        # Write the frame
        profiler.write(frame, request)
    """

    def __init__(
        self,
        profilers: list[Profiler] | None = None,
        loaders: list[Loader] | None = None,
        writers: list[Writer] | None = None,
        available_engines: list[EngineType] | None = None,
    ) -> None:
        self._available_engines = available_engines if available_engines is not None else _detect_available_engines()

        self._profiler_registry = ProfilerRegistry()
        for p in profilers or [RuleBasedEngineProfiler()]:
            self._profiler_registry.register(p)

        self._loader_registry = LoaderRegistry()
        for loader in loaders or [DuckDBLoader(), SparkLoader()]:
            self._loader_registry.register(loader)

        self._writer_registry = WriterRegistry()
        for writer in writers or [DuckDBWriter(), SparkWriter()]:
            self._writer_registry.register(writer)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def recommend(self, request: PipelineRequest) -> ProfilingResult:
        """Run all profilers and return engine recommendations."""
        stamped = dataclasses.replace(request, available_engines=self._available_engines)
        results = self._profiler_registry.run(stamped)
        if not results:
            raise RuntimeError("No profiler could handle this request.")
        merged = {}
        for result in results:
            for rec in result.recommendations:
                if rec.engine not in merged or rec.confidence > merged[rec.engine].confidence:
                    merged[rec.engine] = rec
        best = results[0]
        best.recommendations = sorted(merged.values(), key=lambda r: r.confidence, reverse=True)
        return best

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        """Profile and load the source dataset into a narwhals LazyFrame."""
        engine = self._resolve_engine(request)
        loader = self._loader_registry.get(engine)
        if not loader.can_load(request):
            raise NotImplementedError(f"{engine.value} loader does not support format {request.source.format.value}")
        return loader.load(request)

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        """Write a narwhals LazyFrame to the destination specified in request."""
        if request.destination is None:
            raise ValueError("PipelineRequest has no destination set.")
        engine = self._resolve_engine(request)
        writer = self._writer_registry.get(engine)
        if not writer.can_write(request):
            raise NotImplementedError(f"{engine.value} writer does not support format {request.destination.format.value}")
        writer.write(frame, request)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _resolve_engine(self, request: PipelineRequest) -> EngineType:
        profiling = self.recommend(request)
        if not profiling.recommendations:
            raise RuntimeError(
                f"No available engine can handle this request. "
                f"Install one of: {[e.value for e in EngineType]}"
            )
        return profiling.recommendations[0].engine
