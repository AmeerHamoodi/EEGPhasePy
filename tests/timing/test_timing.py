import time

import pytest

from EEGPhasePy.utils.timing import precise_sleep


@pytest.mark.parametrize("attempt", range(50))
def test_precise_sleep_waits_for_50_milliseconds(attempt):
    requested_duration = 0.05

    start_time = time.perf_counter()
    precise_sleep(requested_duration)
    elapsed_time = time.perf_counter() - start_time

    assert elapsed_time == pytest.approx(requested_duration, abs=0.001)


def test_precise_sleep_returns_immediately_for_zero_duration():
    start_time = time.perf_counter()
    precise_sleep(0)
    elapsed_time = time.perf_counter() - start_time

    assert elapsed_time < 0.001


def test_precise_sleep_rejects_negative_duration():
    with pytest.raises(ValueError, match="duration must be non-negative"):
        precise_sleep(-0.001)
