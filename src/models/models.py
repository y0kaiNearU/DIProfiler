from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


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


class OperationType(Enum):
    FILTER = "filter"
    JOIN = "join"
    AGGREGATE = "aggregate"
    SORT = "sort"
    WINDOW = "window"


@dataclass
class FileSource:
    """Source data from a file system."""

    path: str
    format: FileFormat


@dataclass
class DatabaseSource:
    """Source data from a database."""

    connection_string: str
    table_name: str
    database_type: str  # "postgresql", "mysql", "snowflake", etc.
    schema: str = "public"
    query: Optional[str] = None  # DuckDB SQL run against the attached DB as '_db'; overrides table_name


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


@dataclass
class EngineRecommendation:
    engine: EngineType
    confidence: float
    reasoning: str


@dataclass
class ProfilingResult:
    request: PipelineRequest
    recommendations: list[EngineRecommendation] = field(default_factory=list)

    @property
    def best(self) -> Optional[EngineRecommendation]:
        if not self.recommendations:
            return None
        return max(self.recommendations, key=lambda r: r.confidence)
