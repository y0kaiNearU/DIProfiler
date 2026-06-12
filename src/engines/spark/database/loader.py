from __future__ import annotations

from typing import Any

import narwhals as nw

from engines.spark.database.common import SUPPORTED_DATABASES, jdbc_reader
from models.models import DatabaseSource


def load(spark: Any, src: DatabaseSource) -> nw.LazyFrame:
    try:
        native = jdbc_reader(spark, src).load()
    except Exception as e:
        location = "custom query" if src.query else f"'{src.schema}.{src.table_name}'"
        raise RuntimeError(
            f"Spark failed to read from {src.database_type} {location}: {e}"
        ) from e
    return nw.from_native(native)
