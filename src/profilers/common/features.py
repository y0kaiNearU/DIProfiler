from __future__ import annotations

import math

from models.models import (
    DatabaseSource,
    FileFormat,
    FileSource,
    OperationType,
    PipelineRequest,
)

_FORMATS = list(FileFormat)
_DB_TYPES = ["postgresql", "mysql", "sqlite", "oracle"]
_OPERATIONS = list(OperationType)

FEATURE_NAMES: list[str] = [
    "log_size_bytes",
    "log_row_count",
    "num_columns",
    *[f"src_fmt_{f.value}" for f in _FORMATS],
    "src_is_file",
    "src_is_database",
    *[f"src_db_{t}" for t in _DB_TYPES],
    "has_destination",
    *[f"dst_fmt_{f.value}" for f in _FORMATS],
    "dst_is_file",
    "dst_is_database",
    *[f"dst_db_{t}" for t in _DB_TYPES],
    *[f"op_{o.value}" for o in _OPERATIONS],
]


def extract(request: PipelineRequest) -> list[float]:
    src_info = request.source
    src = src_info.source

    feats: list[float] = []

    feats.append(math.log1p(src_info.size_bytes) if src_info.size_bytes else 0.0)
    feats.append(math.log1p(src_info.row_count) if src_info.row_count else 0.0)
    feats.append(float(src_info.num_columns or 0))

    src_fmt = src.format if isinstance(src, FileSource) else None
    feats.extend(float(src_fmt == f) for f in _FORMATS)

    feats.append(float(isinstance(src, FileSource)))
    feats.append(float(isinstance(src, DatabaseSource)))

    src_db = src.database_type if isinstance(src, DatabaseSource) else None
    feats.extend(float(src_db == t) for t in _DB_TYPES)

    dst_info = request.destination
    feats.append(float(dst_info is not None))

    dst = dst_info.source if dst_info is not None else None
    dst_fmt = dst.format if isinstance(dst, FileSource) else None
    feats.extend(float(dst_fmt == f) for f in _FORMATS)
    feats.append(float(isinstance(dst, FileSource)))
    feats.append(float(isinstance(dst, DatabaseSource)))
    dst_db = dst.database_type if isinstance(dst, DatabaseSource) else None
    feats.extend(float(dst_db == t) for t in _DB_TYPES)

    ops = set(request.operations)
    feats.extend(float(o in ops) for o in _OPERATIONS)

    return feats
