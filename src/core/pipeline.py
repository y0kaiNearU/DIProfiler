from __future__ import annotations

import dataclasses
import importlib.util
from typing import Any, Callable

from core.capability_config import build_default_capabilities, build_required_capabilities
from core.capabilities import CapabilityRegistry
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
    Profiles datasets and provides capability-aware engine recommendations.

    Only recommends engines that can actually handle the source and destination formats.

    Usage:
        profiler = DIProfiler()
        result = profiler.recommend(request)  # Only viable engines recommended
        engine = result.recommendations[0].engine

        # Use FrameLoader/FrameWriter for I/O
        from core.io import FrameLoader, FrameWriter
        loader = FrameLoader(engine)
        frame = loader.load(request)
        frame = frame.filter(...)  # narwhals transforms
        writer = FrameWriter(engine)
        writer.write(frame, request)
    """

    def __init__(
        self,
        profilers: list[Profiler] | None = None,
        loaders: list[Loader] | None = None,
        writers: list[Writer] | None = None,
        capability_registry: CapabilityRegistry | None = None,
        available_engines: list[EngineType] | None = None,
        engine_factories: dict[EngineType, Callable[[], Any]] | None = None,
    ) -> None:
        self._available_engines = available_engines if available_engines is not None else _detect_available_engines()

        self._profiler_registry = ProfilerRegistry()
        for p in profilers or [RuleBasedEngineProfiler()]:
            self._profiler_registry.register(p)

        factories = engine_factories or {}
        duckdb_factory = factories.get(EngineType.DUCKDB)
        spark_factory = factories.get(EngineType.SPARK)

        self._loader_registry = LoaderRegistry()
        self._loader_registry.register(*(loaders or [
            DuckDBLoader(factory=duckdb_factory),
            SparkLoader(factory=spark_factory),
        ]))

        self._writer_registry = WriterRegistry()
        self._writer_registry.register(*(writers or [
            DuckDBWriter(factory=duckdb_factory),
            SparkWriter(factory=spark_factory),
        ]))

        self._capabilities = capability_registry or build_default_capabilities()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def recommend(self, request: PipelineRequest) -> ProfilingResult:
        """Run all profilers and return engine recommendations, filtered by capabilities."""
        stamped = dataclasses.replace(request, available_engines=self._available_engines)
        results = self._profiler_registry.run(stamped)
        if not results:
            raise RuntimeError("No profiler could handle this request.")

        # Build required capabilities from request
        required_caps = build_required_capabilities(request)

        # Filter recommendations to only engines that satisfy required capabilities
        for result in results:
            viable_recs = [
                rec for rec in result.recommendations
                if self._capabilities.can_handle(rec.engine, required_caps)
            ]
            result.recommendations = viable_recs

        # Merge recommendations from all profilers
        merged = {}
        for result in results:
            for rec in result.recommendations:
                if rec.engine not in merged or rec.confidence > merged[rec.engine].confidence:
                    merged[rec.engine] = rec
        best = results[0]
        best.recommendations = sorted(merged.values(), key=lambda r: r.confidence, reverse=True)

        if not best.recommendations:
            raise RuntimeError(
                "No engine can handle this request. "
                "No available engine supports the required capabilities: "
                f"source format {request.source.format.value}, "
                f"{'destination format ' + request.destination.format.value if request.destination else 'no destination'}."
            )
        return best

