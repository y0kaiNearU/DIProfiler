from engine_selection.loader import Loader
from engine_selection.writer import Writer
from models.models import EngineType, PipelineRequest


class LoaderRegistry:

    def __init__(self) -> None:
        self._loaders: list[Loader] = []

    def register(self, *loaders: Loader) -> None:
        self._loaders.extend(loaders)

    def unregister(self, engine: EngineType) -> None:
        self._loaders = [loader for loader in self._loaders if loader.engine != engine]

    def resolve(self, engine: EngineType, request: PipelineRequest) -> Loader:
        for loader in self._loaders:
            if loader.engine == engine and loader.can_load(request):
                return loader
        raise KeyError(f"No loader for engine '{engine.value}' that can handle this request.")

    @property
    def engines(self) -> list[EngineType]:
        return list({loader.engine for loader in self._loaders})


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
