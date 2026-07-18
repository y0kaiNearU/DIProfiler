from core.profiler import Profiler
from models.models import PipelineRequest, ProfilingResult


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

    @property
    def profilers(self) -> list[Profiler]:
        return list(self._profilers.values())

    def run(self, request: PipelineRequest) -> list[ProfilingResult]:
        return [p.profile(request) for p in self._profilers.values() if p.can_handle(request)]

    def run_one(self, name: str, request: PipelineRequest) -> ProfilingResult:
        return self.get(name).profile(request)
