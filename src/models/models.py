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
