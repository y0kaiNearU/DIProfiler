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


    if sys.platform != "win32":
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            return None
    return None
