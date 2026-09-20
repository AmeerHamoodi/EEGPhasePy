import time


def precise_sleep(duration):
    """
    Sleep for a low-jitter duration in seconds.

    This helper reduces wake-up variability compared to a single
    ``time.sleep()`` call on general-purpose operating systems. It uses a
    deadline-based hybrid strategy: sleep in larger chunks while there is
    still time remaining, then do a short busy-wait for the final few
    milliseconds to more closely match the target deadline.

    Parameters
    ----------
    duration : float
        Time to wait in seconds. Must be non-negative.

    Notes
    -----
    This is not a real-time operating system guarantee. It reduces jitter for
    trigger timing on non-RTOS systems, but OS scheduling and hardware latency
    can still affect the final delivery time.
    """
    if duration < 0:
        raise ValueError("duration must be non-negative")

    if duration == 0:
        return

    end_time = time.perf_counter() + duration
    busy_wait_duration = 0.01

    if duration <= 0.05:
        while time.perf_counter() < end_time:
            pass
        return

    while True:
        remaining = end_time - time.perf_counter()
        if remaining <= busy_wait_duration:
            break
        time.sleep(max(0.0, remaining - busy_wait_duration))

    while time.perf_counter() < end_time:
        pass
