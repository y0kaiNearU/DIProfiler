from __future__ import annotations

import dataclasses
import importlib.util
from typing import Any, Callable

from core.profiler import Profiler
from core.registry import ProfilerRegistry
from engine_selection.capability_config import build_default_capabilities, build_required_capabilities
from engine_selection.capabilities import CapabilityRegistry
from engine_selection.loader import Loader
from engine_selection.registry import LoaderRegistry, WriterRegistry
from engine_selection.writer import Writer
from engines.dask.loader import DaskLoader
from engines.dask.writer import DaskWriter
from engines.datafusion.loader import DataFusionLoader
from engines.datafusion.writer import DataFusionWriter
from engines.duckdb.loader import DuckDBLoader
from engines.duckdb.writer import DuckDBWriter
from engines.polars.loader import PolarsLoader
from engines.polars.writer import PolarsWriter
from engines.spark.loader import SparkLoader
from engines.spark.writer import SparkWriter
from models.models import DatabaseSource, DatasetInfo, EngineType, FileSource, PipelineRequest, ProfilingResult
from profilers.common.composite import EnsembleProfiler
from profilers.engine.rule_based_engine_profiler import RuleBasedEngineProfiler

_ENGINE_MODULES: dict[EngineType, str] = {
    EngineType.DUCKDB: "duckdb",
    EngineType.SPARK: "pyspark",
    EngineType.DATAFUSION: "datafusion",
    EngineType.POLARS: "polars",
    EngineType.DASK: "dask",
}


def _detect_available_engines() -> list[EngineType]:
    return [e for e, mod in _ENGINE_MODULES.items() if importlib.util.find_spec(mod) is not None]


def _describe_source(info: DatasetInfo) -> str:
    src = info.source
    if isinstance(src, FileSource):
        return f"{src.format.value} file"
    if isinstance(src, DatabaseSource):
        return f"{src.database_type} database"
    return type(src).__name__


class DIProfiler:
    """
    Profiles datasets and provides capability-aware engine recommendations.

    Only recommends engines that can actually handle the source and destination formats.

    Usage:
        profiler = DIProfiler()
        result = profiler.recommend(request)  # Only viable engines recommended
        engine = result.recommendations[0].engine

        # Use FrameLoader/FrameWriter for I/O
        from engine_selection.io import FrameLoader, FrameWriter
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
        datafusion_factory = factories.get(EngineType.DATAFUSION)
        dask_factory = factories.get(EngineType.DASK)

        self._loader_registry = LoaderRegistry()
        self._loader_registry.register(*(loaders or [
            DuckDBLoader(factory=duckdb_factory),
            SparkLoader(factory=spark_factory),
            DataFusionLoader(factory=datafusion_factory),
            PolarsLoader(),
            DaskLoader(factory=dask_factory),
        ]))

        self._writer_registry = WriterRegistry()
        self._writer_registry.register(*(writers or [
            DuckDBWriter(factory=duckdb_factory),
            SparkWriter(factory=spark_factory),
            DataFusionWriter(factory=datafusion_factory),
            PolarsWriter(),
            DaskWriter(factory=dask_factory),
        ]))

        self._capabilities = capability_registry or build_default_capabilities()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def recommend(self, request: PipelineRequest) -> ProfilingResult:
        """Run all profilers and return engine recommendations, filtered by capabilities."""
        stamped = dataclasses.replace(request, available_engines=self._available_engines)
        required_caps = build_required_capabilities(request)

        ensemble = EnsembleProfiler(
            self._profiler_registry.profilers,
            key_fn=lambda rec: rec.engine,
            mode="max",
            filter_fn=lambda rec: self._capabilities.can_handle(rec.engine, required_caps),
        )

        if not ensemble.can_handle(stamped):
            raise RuntimeError("No profiler could handle this request.")

        result = ensemble.profile(stamped)

        if not result.recommendations:
            destination_desc = (
                f"destination {_describe_source(request.destination)}" if request.destination else "no destination"
            )
            raise RuntimeError(
                "No engine can handle this request. "
                "No available engine supports the required capabilities: "
                f"source {_describe_source(request.source)}, {destination_desc}."
            )
        return result

