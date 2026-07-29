from __future__ import annotations

from engine_selection.capabilities import SupportsDataSource, SupportsFormat
from models.models import FileFormat

# JSON is excluded: modin 0.37's read_json raises internally regardless of
# execution engine (PandasOnPython and PandasOnDask both fail), so only the
# formats verified to actually round-trip are declared here.
CAPABILITIES = [
    SupportsFormat(FileFormat.CSV, "read"),
    SupportsFormat(FileFormat.PARQUET, "read"),
    SupportsFormat(FileFormat.CSV, "write"),
    SupportsFormat(FileFormat.PARQUET, "write"),
    SupportsDataSource("filesystem", "read"),
    SupportsDataSource("filesystem", "write"),
]
