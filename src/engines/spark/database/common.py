from __future__ import annotations

from typing import Any

from models.models import DatabaseSource

SUPPORTED_DATABASES = ("postgresql", "mysql", "oracle")

# MySQL has no schema layer; Oracle and PostgreSQL do
_HAS_SCHEMA = {"postgresql", "oracle"}


def qualified_table(src: DatabaseSource) -> str:
    if src.database_type in _HAS_SCHEMA:
        return f"{src.schema}.{src.table_name}"
    return src.table_name


def jdbc_reader(spark: Any, src: DatabaseSource) -> Any:
    """Build a Spark JDBC DataFrameReader, using query pushdown if set."""
    reader = spark.read.format("jdbc").option("url", src.connection_string)
    if src.query:
        reader = reader.option("query", src.query)
    else:
        reader = reader.option("dbtable", qualified_table(src))
    return reader
