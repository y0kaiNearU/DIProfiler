from __future__ import annotations

from engine_selection.capabilities import SupportsDataSource, SupportsFormat
from models.models import FileFormat

CAPABILITIES = [
    SupportsFormat(FileFormat.CSV, "read"),
    SupportsFormat(FileFormat.PARQUET, "read"),
    SupportsFormat(FileFormat.CSV, "write"),
    SupportsFormat(FileFormat.PARQUET, "write"),
    SupportsDataSource("filesystem", "read"),
    SupportsDataSource("filesystem", "write"),
]
