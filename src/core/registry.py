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
        self._loaders: list[Loader] = []

    def register(self, *loaders: Loader) -> None:
        self._loaders.extend(loaders)

    def unregister(self, engine: EngineType) -> None:
        self._loaders = [l for l in self._loaders if l.engine != engine]

    def resolve(self, engine: EngineType, request: PipelineRequest) -> Loader:
        for loader in self._loaders:
            if loader.engine == engine and loader.can_load(request):
                return loader
        raise KeyError(f"No loader for engine '{engine.value}' that can handle this request.")

    @property
    def engines(self) -> list[EngineType]:
        return list({l.engine for l in self._loaders})


class WriterRegistry:

    def __init__(self) -> None:
        self._writers: list[Writer] = []

    def register(self, *writers: Writer) -> None:
        self._writers.extend(writers)

    def unregister(self, engine: EngineType) -> None:
        self._writers = [w for w in self._writers if w.engine != engine]

    def resolve(self, engine: EngineType, request: PipelineRequest) -> Writer:
        for writer in self._writers:
            if writer.engine == engine and writer.can_write(request):
                return writer
        raise KeyError(f"No writer for engine '{engine.value}' that can handle this request.")

    @property
    def engines(self) -> list[EngineType]:
        return list({w.engine for w in self._writers})
