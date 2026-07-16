from __future__ import annotations

from core.capabilities import SupportsDataSource, SupportsFormat
from models.models import FileFormat

CAPABILITIES = [
    SupportsFormat(FileFormat.CSV, "read"),
    SupportsFormat(FileFormat.PARQUET, "read"),
    SupportsFormat(FileFormat.JSON, "read"),
    SupportsFormat(FileFormat.CSV, "write"),
    SupportsFormat(FileFormat.PARQUET, "write"),
    SupportsFormat(FileFormat.JSON, "write"),
    SupportsDataSource("filesystem", "read"),
    SupportsDataSource("filesystem", "write"),
]
