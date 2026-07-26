from profilers.resource.detect import detect_available_resources


def test_detect_available_resources_returns_positive_cores():
    cores, memory_bytes = detect_available_resources()
    assert isinstance(cores, int)
    assert cores >= 1


def test_detect_available_resources_memory_is_int_or_none():
    _, memory_bytes = detect_available_resources()
    assert memory_bytes is None or (isinstance(memory_bytes, int) and memory_bytes > 0)
