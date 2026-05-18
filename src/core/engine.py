from abc import ABC, abstractmethod
from typing import Any

from models.models import EngineType


class Engine(ABC):
    """Unified API over processing engines (DuckDB, Spark)."""

    @property
    @abstractmethod
    def engine_type(self) -> EngineType: ...

    @abstractmethod
    def read(self, path: str, **kwargs) -> Any:
        """Read a dataset from path into an engine-native representation."""

    @abstractmethod
    def filter(self, dataset: Any, condition: str) -> Any:
        """Filter rows by a SQL-style condition string."""

    @abstractmethod
    def select(self, dataset: Any, columns: list[str]) -> Any:
        """Project a subset of columns."""

    @abstractmethod
    def join(self, left: Any, right: Any, on: str | list[str], how: str = "inner") -> Any:
        """Join two datasets."""

    @abstractmethod
    def aggregate(self, dataset: Any, group_by: list[str], agg_exprs: dict[str, str]) -> Any:
        """
        Group and aggregate.
        agg_exprs maps output column name → expression, e.g. {"total": "sum(amount)"}
        """

    @abstractmethod
    def write(self, dataset: Any, path: str, **kwargs) -> None:
        """Write dataset to path."""

    @abstractmethod
    def sql(self, query: str) -> Any:
        """Run an arbitrary SQL query and return the result."""
