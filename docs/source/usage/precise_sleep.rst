Precise sleeping
================

When scheduling a real-time stimulus, the delay between deciding to trigger
and sending the trigger should be as consistent as possible. The
:py:func:`~EEGPhasePy.utils.timing.precise_sleep` helper waits for a requested
number of seconds using a combination of regular sleeping and a short final
busy-wait.

Using ``precise_sleep``
-----------------------

Import the function from the timing utility module. Durations are specified in
seconds, so ``0.05`` requests a 50 millisecond wait:

.. code-block:: python
   :caption: Wait for 50 milliseconds

   from EEGPhasePy.utils.timing import precise_sleep

   precise_sleep(0.05)

The function accepts non-negative durations. A duration of ``0`` returns
immediately, while a negative duration raises ``ValueError``.

Measuring the wait
------------------

You can measure the actual delay with Python's monotonic, high-resolution
``time.perf_counter`` clock:

.. code-block:: python
   :caption: Measure a precise sleep

   import time

   from EEGPhasePy.utils.timing import precise_sleep

   requested_duration = 0.05
   start_time = time.perf_counter()
   precise_sleep(requested_duration)
   elapsed_time = time.perf_counter() - start_time

   print(f"Requested: {requested_duration * 1000:.1f} ms")
   print(f"Elapsed: {elapsed_time * 1000:.1f} ms")

Testing timing reliability
--------------------------

Timing can vary because of operating system scheduling and other processes
running on the computer. To evaluate reliability, repeat the measurement over
multiple trials and inspect the difference between the requested and measured
durations:

.. code-block:: python
   :caption: Measure timing over 20 trials

   import time

   from EEGPhasePy.utils.timing import precise_sleep

   requested_duration = 0.05
   tolerance = 0.001
   results = []

   for _ in range(20):
       start_time = time.perf_counter()
       precise_sleep(requested_duration)
       elapsed_time = time.perf_counter() - start_time
       results.append(elapsed_time)

   successful_trials = sum(
       abs(elapsed_time - requested_duration) <= tolerance
       for elapsed_time in results
   )

   print(f"Trials within 1 ms: {successful_trials}/{len(results)}")

``precise_sleep`` reduces timing jitter on general-purpose operating systems,
but it does not provide a hard real-time guarantee. For experiments that
require precise stimulus delivery, measure the complete hardware pipeline,
including amplifier, operating system, and stimulator delays.
