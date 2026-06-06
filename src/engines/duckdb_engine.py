import duckdb

from core.engine import Engine
from models.models import EngineType, FileFormat


class DuckDBEngine(Engine):

    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None) -> None:
        self._conn = connection or duckdb.connect()

    @property
    def engine_type(self) -> EngineType:
        return EngineType.DUCKDB

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #

    def read(self, path: str, format: FileFormat | None = None, **kwargs) -> duckdb.DuckDBPyRelation:
        fmt = format or self._infer_format(path)
        match fmt:
            case FileFormat.CSV:
                return self._conn.read_csv(path, **kwargs)
            case FileFormat.PARQUET:
                return self._conn.read_parquet(path, **kwargs)
            case FileFormat.JSON:
                return self._conn.read_json(path, **kwargs)
            case _:
                raise NotImplementedError(f"DuckDB engine does not support format {fmt}")

    def write(self, dataset: duckdb.DuckDBPyRelation, path: str, format: FileFormat | None = None, **kwargs) -> None:
        fmt = format or self._infer_format(path)
        match fmt:
            case FileFormat.CSV:
                dataset.write_csv(path, **kwargs)
            case FileFormat.PARQUET:
                dataset.write_parquet(path, **kwargs)
            case _:
                raise NotImplementedError(f"DuckDB engine cannot write format {fmt}")

    # ------------------------------------------------------------------ #
    # Transformations                                                       #
    # ------------------------------------------------------------------ #

    def filter(self, dataset: duckdb.DuckDBPyRelation, condition: str) -> duckdb.DuckDBPyRelation:
        return dataset.filter(condition)

    def select(self, dataset: duckdb.DuckDBPyRelation, columns: list[str]) -> duckdb.DuckDBPyRelation:
        return dataset.select(", ".join(columns))

    def join(
        self,
        left: duckdb.DuckDBPyRelation,
        right: duckdb.DuckDBPyRelation,
        on: str | list[str],
        how: str = "inner",
    ) -> duckdb.DuckDBPyRelation:
        condition = " AND ".join(on) if isinstance(on, list) else on
        return left.join(right, condition, how)

    def aggregate(
        self,
        dataset: duckdb.DuckDBPyRelation,
        group_by: list[str],
        agg_exprs: dict[str, str],
    ) -> duckdb.DuckDBPyRelation:
        named_aggs = ", ".join(f"{expr} AS {alias}" for alias, expr in agg_exprs.items())
        full_expr = ", ".join(group_by + [named_aggs]) if group_by else named_aggs
        group_expr = ", ".join(group_by) if group_by else None
        return dataset.aggregate(full_expr, group_expr)

    def sql(self, query: str) -> duckdb.DuckDBPyRelation:
        return self._conn.sql(query)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_format(path: str) -> FileFormat:
        suffix = path.rsplit(".", 1)[-1].lower()
        mapping = {
            "csv": FileFormat.CSV,
            "parquet": FileFormat.PARQUET,
            "json": FileFormat.JSON,
            "jsonl": FileFormat.JSON,
        }
        if suffix not in mapping:
            raise ValueError(f"Cannot infer file format from extension '.{suffix}'")
        return mapping[suffix]
