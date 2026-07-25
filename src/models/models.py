from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Union, runtime_checkable


class FileFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    ORC = "orc"
    DELTA = "delta"
    ICEBERG = "iceberg"


class EngineType(Enum):
    DUCKDB = "duckdb"
    SPARK = "spark"
    POLARS = "polars"
    DASK = "dask"
    PANDAS = "pandas"


class OperationType(Enum):
    FILTER = "filter"
    JOIN = "join"
    AGGREGATE = "aggregate"
    SORT = "sort"
    WINDOW = "window"


class WriteMode(Enum):
    OVERWRITE = "overwrite"
    APPEND = "append"
    IGNORE = "ignore"
    ERROR = "error"


@dataclass
class FileSource:
    """Source data from a file system."""

    path: str
    format: FileFormat
    write_mode: WriteMode = WriteMode.OVERWRITE


@dataclass
class DatabaseSource:
    """Source data from a database."""

    connection_string: str
    table_name: str
    database_type: str  # "postgresql", "mysql", "snowflake", etc.
    schema: str = "public"
    queries: dict[EngineType, str] = field(default_factory=dict)  # engine-specific SQL, overrides table_name
    write_mode: WriteMode = WriteMode.OVERWRITE


DatasetSource = Union[FileSource, DatabaseSource]


@dataclass
class DatasetInfo:
    source: DatasetSource
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    num_columns: Optional[int] = None
    schema: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineRequest:
    source: DatasetInfo
    required_engine: Optional[EngineType] = None
    available_engines: list["EngineType"] = field(default_factory=lambda: list(EngineType))
    destination: Optional[DatasetInfo] = None
    operations: list[OperationType] = field(default_factory=list)
    available_cores: Optional[int] = None
    available_memory_bytes: Optional[int] = None


@runtime_checkable
class Recommendation(Protocol):
    """Structural contract every profiler recommendation must satisfy."""

    confidence: float
    reasoning: str


@dataclass
class EngineRecommendation:
    engine: EngineType
    confidence: float
    reasoning: str


@dataclass
class PartitionRecommendation:
    column: str
    confidence: float
    reasoning: str


@dataclass
class ResourceRecommendation:
    cores: int
    memory_bytes: int
    confidence: float
    reasoning: str


@dataclass
class ProfilingResult[R: Recommendation]:
    request: PipelineRequest
    recommendations: list[R] = field(default_factory=list)

    @property
    def best(self) -> Optional[R]:
        if not self.recommendations:
            return None
        return max(self.recommendations, key=lambda r: r.confidence)
