from __future__ import annotations

import narwhals as nw

from core.loader import Loader
from core.registry import LoaderRegistry, WriterRegistry
from core.writer import Writer
from engines.duckdb.loader import DuckDBLoader
from engines.duckdb.writer import DuckDBWriter
from engines.spark.loader import SparkLoader
from engines.spark.writer import SparkWriter
from models.models import EngineType, PipelineRequest


class FrameLoader:
    """Unified API for loading data into narwhals LazyFrames with automatic engine handling."""

    def __init__(self, engine: EngineType, loaders: list[Loader] | None = None) -> None:
        self.engine = engine
        self._registry = LoaderRegistry()
        for loader in loaders or [DuckDBLoader(), SparkLoader()]:
            self._registry.register(loader)

    def load(self, request: PipelineRequest) -> nw.LazyFrame:
        """Load source dataset into a narwhals LazyFrame."""
        loader = self._registry.get(self.engine)
        if not loader.can_load(request):
            raise NotImplementedError(f"{self.engine.value} loader does not support format {request.source.format.value}")
        return loader.load(request)


class FrameWriter:
    """Unified API for writing narwhals LazyFrames with automatic engine handling."""

    def __init__(self, engine: EngineType, writers: list[Writer] | None = None) -> None:
        self.engine = engine
        self._registry = WriterRegistry()
        for writer in writers or [DuckDBWriter(), SparkWriter()]:
            self._registry.register(writer)

    def write(self, frame: nw.LazyFrame, request: PipelineRequest) -> None:
        """Write a narwhals LazyFrame to the destination specified in request."""
        if request.destination is None:
            raise ValueError("PipelineRequest has no destination set.")
        writer = self._registry.get(self.engine)
        if not writer.can_write(request):
            raise NotImplementedError(f"{self.engine.value} writer does not support format {request.destination.format.value}")
        writer.write(frame, request)
