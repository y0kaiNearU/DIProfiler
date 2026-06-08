from core.loader import Loader
from core.profiler import Profiler
from core.writer import Writer
from models.models import EngineType, PipelineRequest, ProfilingResult


class ProfilerRegistry:

    def __init__(self) -> None:
        self._profilers: dict[str, Profiler] = {}

    def register(self, profiler: Profiler) -> None:
        if profiler.name in self._profilers:
            raise ValueError(f"Profiler '{profiler.name}' is already registered.")
        self._profilers[profiler.name] = profiler

    def unregister(self, name: str) -> None:
        self._profilers.pop(name, None)

    def get(self, name: str) -> Profiler:
        if name not in self._profilers:
            raise KeyError(f"No profiler named '{name}'.")
        return self._profilers[name]

    @property
    def names(self) -> list[str]:
        return list(self._profilers)

    def run(self, request: PipelineRequest) -> list[ProfilingResult]:
        return [p.profile(request) for p in self._profilers.values() if p.can_handle(request)]

    def run_one(self, name: str, request: PipelineRequest) -> ProfilingResult:
        return self.get(name).profile(request)


class LoaderRegistry:

    def __init__(self) -> None:
        self._loaders: dict[EngineType, Loader] = {}

    def register(self, loader: Loader) -> None:
        if loader.engine in self._loaders:
            raise ValueError(f"Loader for '{loader.engine.value}' is already registered.")
        self._loaders[loader.engine] = loader

    def unregister(self, engine: EngineType) -> None:
        self._loaders.pop(engine, None)

    def get(self, engine: EngineType) -> Loader:
        if engine not in self._loaders:
            raise KeyError(f"No loader registered for engine '{engine.value}'.")
        return self._loaders[engine]

    @property
    def engines(self) -> list[EngineType]:
        return list(self._loaders)


class WriterRegistry:

    def __init__(self) -> None:
        self._writers: dict[EngineType, Writer] = {}

    def register(self, writer: Writer) -> None:
        if writer.engine in self._writers:
            raise ValueError(f"Writer for '{writer.engine.value}' is already registered.")
        self._writers[writer.engine] = writer

    def unregister(self, engine: EngineType) -> None:
        self._writers.pop(engine, None)

    def get(self, engine: EngineType) -> Writer:
        if engine not in self._writers:
            raise KeyError(f"No writer registered for engine '{engine.value}'.")
        return self._writers[engine]

    @property
    def engines(self) -> list[EngineType]:
        return list(self._writers)
