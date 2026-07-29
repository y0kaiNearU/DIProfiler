import pytest

from models.models import DatasetInfo, FileFormat, FileSource, PipelineRequest
from profilers.resource.rule_based_resource_profiler import RuleBasedResourceProfiler

_GB = 1024 ** 3


def _req(size_bytes=None, available_cores=None, available_memory_bytes=None):
    return PipelineRequest(
        source=DatasetInfo(source=FileSource(path="x", format=FileFormat.CSV), size_bytes=size_bytes),
        available_cores=available_cores,
        available_memory_bytes=available_memory_bytes,
    )


class TestRuleBasedResourceProfiler:
    def setup_method(self):
        self.profiler = RuleBasedResourceProfiler()

    def test_name(self):
        assert self.profiler.name == "rule_based_resource_profiler"

    def test_can_handle_requires_both_budgets(self):
        assert self.profiler.can_handle(_req(available_cores=8, available_memory_bytes=16 * _GB))
        assert not self.profiler.can_handle(_req(available_cores=8))
        assert not self.profiler.can_handle(_req(available_memory_bytes=16 * _GB))
        assert not self.profiler.can_handle(_req())

    def test_scales_cores_with_dataset_size(self):
        small = self.profiler.profile(
            _req(size_bytes=500 * 1024 ** 2, available_cores=16, available_memory_bytes=64 * _GB)
        )
        large = self.profiler.profile(_req(size_bytes=8 * _GB, available_cores=16, available_memory_bytes=64 * _GB))
        assert small.best.cores < large.best.cores

    def test_cores_never_exceed_available(self):
        result = self.profiler.profile(_req(size_bytes=500 * _GB, available_cores=16, available_memory_bytes=64 * _GB))
        assert result.best.cores <= 16

    def test_memory_never_exceeds_available(self):
        result = self.profiler.profile(_req(size_bytes=500 * _GB, available_cores=16, available_memory_bytes=64 * _GB))
        assert result.best.memory_bytes <= 64 * _GB

    def test_at_least_one_core_recommended(self):
        result = self.profiler.profile(_req(size_bytes=1024, available_cores=16, available_memory_bytes=64 * _GB))
        assert result.best.cores >= 1

    def test_unknown_size_uses_fallback_fraction(self):
        result = self.profiler.profile(_req(available_cores=16, available_memory_bytes=64 * _GB))
        assert result.best.cores == 4
        assert result.best.memory_bytes == pytest.approx(16 * _GB, rel=1e-6)

    def test_unknown_size_has_lower_confidence(self):
        known = self.profiler.profile(_req(size_bytes=1 * _GB, available_cores=16, available_memory_bytes=64 * _GB))
        unknown = self.profiler.profile(_req(available_cores=16, available_memory_bytes=64 * _GB))
        assert unknown.best.confidence < known.best.confidence

    def test_returns_single_recommendation(self):
        result = self.profiler.profile(_req(size_bytes=1 * _GB, available_cores=16, available_memory_bytes=64 * _GB))
        assert len(result.recommendations) == 1
