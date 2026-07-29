from __future__ import annotations

import os
import sys


def detect_available_resources() -> tuple[int, int | None]:
    cores = os.cpu_count() or 1
    return cores, _detect_memory_bytes()


def _detect_memory_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except ImportError:
        pass

    # os.sysconf is POSIX-only; the sys.platform guard (rather than a bare
    # try/except AttributeError) lets mypy type-check this branch on POSIX
    # and skip it as unreachable on Windows, without needing a type: ignore
    # that would be correct on one platform and stale on the other.
    if sys.platform != "win32":
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            return None
    return None
