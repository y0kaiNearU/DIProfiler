from __future__ import annotations

import narwhals as nw

from engines.spark.database.common import SUPPORTED_DATABASES, qualified_table
from models.models import DatabaseSource


def write(frame: nw.LazyFrame, dest: DatabaseSource) -> None:
    try:
        nw.to_native(frame).write.format("jdbc").option("url", dest.connection_string).option(
            "dbtable", qualified_table(dest)
        ).mode(dest.write_mode.value).save()
    except Exception as e:
        raise RuntimeError(
            f"Spark failed to write to {dest.database_type} "
            f"'{dest.schema}.{dest.table_name}': {e}"
        ) from e
