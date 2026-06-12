from __future__ import annotations

from typing import Any

import narwhals as nw

from models.models import DatabaseSource

SUPPORTED_DATABASES = ("postgresql", "mysql", "oracle")


def load(spark: Any, src: DatabaseSource) -> nw.LazyFrame:
    if src.database_type not in SUPPORTED_DATABASES:
        raise NotImplementedError(f"Spark database loader does not support {src.database_type}")
    native = spark.read.format("jdbc").option("url", src.connection_string).option(
        "dbtable", src.table_name
    ).load()
    return nw.from_native(native)
