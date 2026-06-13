from __future__ import annotations

from typing import Any

import narwhals as nw

from models.models import FileFormat, FileSource

SUPPORTED_FORMATS = (FileFormat.CSV, FileFormat.PARQUET, FileFormat.JSON)


def load(ctx: Any, src: FileSource) -> nw.LazyFrame:
    try:
        match src.format:
            case FileFormat.CSV:
                native = ctx.read_csv(src.path)
            case FileFormat.PARQUET:
                native = ctx.read_parquet(src.path)
            case FileFormat.JSON:
                native = ctx.read_json(src.path)
            case _:
                raise NotImplementedError(f"DataFusion file loader does not support {src.format}")
    except NotImplementedError:
        raise
    except Exception as e:
        raise RuntimeError(f"DataFusion failed to read {src.format.value} from '{src.path}': {e}") from e

    return nw.from_native(native)
