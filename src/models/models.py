from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FileFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    ORC = "orc"
    DELTA = "delta"


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
class DatasetInfo:
    path: str
    format: FileFormat
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    num_columns: Optional[int] = None
    schema: dict[str, str] = field(default_factory=dict)


@dataclass
class ProfilingRequest:
    dataset: DatasetInfo
    operations: list[OperationType] = field(default_factory=list)
    required_engine: Optional[EngineType] = None


@dataclass
class EngineRecommendation:
    engine: EngineType
    confidence: float
    reasoning: str


@dataclass
class ProfilingResult:
    request: ProfilingRequest
    recommendations: list[EngineRecommendation] = field(default_factory=list)

    @property
    def best(self) -> Optional[EngineRecommendation]:
        if not self.recommendations:
            return None
        return max(self.recommendations, key=lambda r: r.confidence)
