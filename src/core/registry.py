from core.profiler import Profiler
from models.models import ProfilingRequest, ProfilingResult


class ProfilerRegistry:
    """Holds registered profilers and runs them against a request."""

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

    def run(self, request: ProfilingRequest) -> list[ProfilingResult]:
        """Run all applicable profilers and return their results."""
        results = []
        for profiler in self._profilers.values():
            if profiler.can_handle(request):
                results.append(profiler.profile(request))
        return results

    def run_one(self, name: str, request: ProfilingRequest) -> ProfilingResult:
        return self.get(name).profile(request)
