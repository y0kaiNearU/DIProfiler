from __future__ import annotations

from typing import Any

import narwhals as nw

from models.models import DatabaseSource

SUPPORTED_DATABASES = ("postgresql", "mysql", "oracle")


def write(frame: nw.LazyFrame, dest: DatabaseSource) -> None:
    if dest.database_type not in SUPPORTED_DATABASES:
        raise NotImplementedError(f"Spark database writer does not support {dest.database_type}")
    nw.to_native(frame).write.format("jdbc").option("url", dest.connection_string).option(
        "dbtable", dest.table_name
    ).mode("overwrite").save()
