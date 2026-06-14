import pytest

from core.capabilities import CapabilityRegistry, SupportsDataSource, SupportsFormat
from core.capability_config import build_default_capabilities, build_required_capabilities
from models.models import (
    DatasetInfo,
    DatabaseSource,
    EngineType,
    FileFormat,
    FileSource,
    PipelineRequest,
)


class TestSupportsFormat:
    def test_matches_same_format_and_direction(self):
        cap = SupportsFormat(FileFormat.CSV, "read")
        req = SupportsFormat(FileFormat.CSV, "read")
        assert cap.matches(req)

    def test_does_not_match_different_format(self):
        cap = SupportsFormat(FileFormat.CSV, "read")
        req = SupportsFormat(FileFormat.PARQUET, "read")
        assert not cap.matches(req)

    def test_does_not_match_different_direction(self):
        cap = SupportsFormat(FileFormat.CSV, "read")
        req = SupportsFormat(FileFormat.CSV, "write")
        assert not cap.matches(req)

    def test_does_not_match_different_type(self):
        cap = SupportsFormat(FileFormat.CSV, "read")
        req = SupportsDataSource("filesystem", "read")
        assert not cap.matches(req)


class TestSupportsDataSource:
    def test_matches_same_source_and_direction(self):
        cap = SupportsDataSource("filesystem", "read")
        req = SupportsDataSource("filesystem", "read")
        assert cap.matches(req)

    def test_does_not_match_different_source(self):
        cap = SupportsDataSource("filesystem", "read")
        req = SupportsDataSource("s3", "read")
        assert not cap.matches(req)

    def test_does_not_match_different_direction(self):
        cap = SupportsDataSource("filesystem", "read")
        req = SupportsDataSource("filesystem", "write")
        assert not cap.matches(req)


class TestCapabilityRegistry:
    def setup_method(self):
        self.registry = CapabilityRegistry()
        self.registry.register(
            EngineType.DUCKDB,
            SupportsFormat(FileFormat.CSV, "read"),
            SupportsFormat(FileFormat.PARQUET, "read"),
            SupportsDataSource("filesystem", "read"),
        )

    def test_can_handle_when_all_caps_satisfied(self):
        required = [SupportsFormat(FileFormat.CSV, "read")]
        assert self.registry.can_handle(EngineType.DUCKDB, required)

    def test_can_handle_multiple_requirements(self):
        required = [
            SupportsFormat(FileFormat.CSV, "read"),
            SupportsDataSource("filesystem", "read"),
        ]
        assert self.registry.can_handle(EngineType.DUCKDB, required)

    def test_cannot_handle_unsatisfied_requirement(self):
        required = [SupportsFormat(FileFormat.ORC, "read")]
        assert not self.registry.can_handle(EngineType.DUCKDB, required)

    def test_cannot_handle_unknown_engine(self):
        required = [SupportsFormat(FileFormat.CSV, "read")]
        assert not self.registry.can_handle(EngineType.SPARK, required)

    def test_empty_requirements_always_satisfied(self):
        assert self.registry.can_handle(EngineType.DUCKDB, [])

    def test_get_returns_registered_caps(self):
        caps = self.registry.get(EngineType.DUCKDB)
        assert SupportsFormat(FileFormat.CSV, "read") in caps

    def test_get_unknown_engine_returns_empty_set(self):
        assert self.registry.get(EngineType.SPARK) == set()


class TestBuildDefaultCapabilities:
    def setup_method(self):
        self.caps = build_default_capabilities()

    def test_duckdb_reads_csv(self):
        assert self.caps.can_handle(EngineType.DUCKDB, [SupportsFormat(FileFormat.CSV, "read")])

    def test_duckdb_does_not_read_orc(self):
        assert not self.caps.can_handle(EngineType.DUCKDB, [SupportsFormat(FileFormat.ORC, "read")])

    def test_spark_reads_orc(self):
        assert self.caps.can_handle(EngineType.SPARK, [SupportsFormat(FileFormat.ORC, "read")])

    def test_spark_reads_delta(self):
        assert self.caps.can_handle(EngineType.SPARK, [SupportsFormat(FileFormat.DELTA, "read")])

    def test_datafusion_reads_parquet(self):
        assert self.caps.can_handle(EngineType.DATAFUSION, [SupportsFormat(FileFormat.PARQUET, "read")])

    def test_duckdb_reads_postgresql(self):
        assert self.caps.can_handle(EngineType.DUCKDB, [SupportsDataSource("postgresql", "read")])

    def test_datafusion_does_not_read_postgresql(self):
        assert not self.caps.can_handle(EngineType.DATAFUSION, [SupportsDataSource("postgresql", "read")])

    def test_spark_reads_oracle(self):
        assert self.caps.can_handle(EngineType.SPARK, [SupportsDataSource("oracle", "read")])


class TestBuildRequiredCapabilities:
    def _file_req(self, fmt, dst_fmt=None):
        src = DatasetInfo(source=FileSource(path="x", format=fmt))
        dst = DatasetInfo(source=FileSource(path="out", format=dst_fmt)) if dst_fmt else None
        return PipelineRequest(source=src, destination=dst)

    def _db_req(self, db_type, dst_db_type=None):
        src = DatasetInfo(source=DatabaseSource(
            connection_string="conn", table_name="t", database_type=db_type,
        ))
        dst = None
        if dst_db_type:
            dst = DatasetInfo(source=DatabaseSource(
                connection_string="conn", table_name="t2", database_type=dst_db_type,
            ))
        return PipelineRequest(source=src, destination=dst)

    def test_file_source_requires_read_format(self):
        caps = build_required_capabilities(self._file_req(FileFormat.CSV))
        assert any(
            isinstance(c, SupportsFormat) and c.format == FileFormat.CSV and c.direction == "read"
            for c in caps
        )

    def test_file_source_requires_filesystem_read(self):
        caps = build_required_capabilities(self._file_req(FileFormat.CSV))
        assert any(
            isinstance(c, SupportsDataSource) and c.source_type == "filesystem" and c.direction == "read"
            for c in caps
        )

    def test_file_destination_requires_write_format(self):
        caps = build_required_capabilities(self._file_req(FileFormat.CSV, dst_fmt=FileFormat.PARQUET))
        assert any(
            isinstance(c, SupportsFormat) and c.format == FileFormat.PARQUET and c.direction == "write"
            for c in caps
        )

    def test_db_source_requires_db_read(self):
        caps = build_required_capabilities(self._db_req("postgresql"))
        assert any(
            isinstance(c, SupportsDataSource) and c.source_type == "postgresql" and c.direction == "read"
            for c in caps
        )

    def test_no_destination_no_write_caps(self):
        caps = build_required_capabilities(self._file_req(FileFormat.CSV))
        assert not any(c for c in caps if hasattr(c, "direction") and c.direction == "write")
