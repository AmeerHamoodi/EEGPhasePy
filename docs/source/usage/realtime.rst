Real-time phase estimation
==========================

Performing phase estimation in real time introduces a set of practical engineering challenges that are largely absent from offline analysis.
Even if phase is estimated perfectly, if the surrounding system is not accounted for a large degree of error in estimation will be introduced.
The two most important sources of error are **delay** and **jitter**.

Delay and jitter
-----------------

**Delay** is the time between an event occurring and your system being able to act on it.
There are two critical delay legs:

- **Amplifier → software**: There is always some lag between the moment a sample is acquired and the moment the software receives and processes it.
  This includes transmission time over USB or network, operating system scheduling latency, and any buffering inside the driver or SDK.

- **Software → stimulator**: Once the software decides to deliver a stimulus, a command must be sent to the stimulation device
  (e.g. a TMS coil, a tES device, or an audio/visual output). The stimulator then takes some additional time to actually deliver
  the stimulus. Together these add up to the total output delay.

The combined effect is that by the time a stimulus is actually delivered, the brain has moved on from the phase that was estimated.
If this total delay is not accounted for, stimuli will land at systematically wrong phases.

**Jitter** is trial-to-trial variability in any of these delays. A constant delay is straightforward to compensate for:
you simply estimate the phase at a fixed time in the future rather than at the present moment. Jitter is harder because it
means the compensation must account for a range of possible delays rather than a single value. Sources of jitter include:

- Operating system scheduling: most general-purpose operating systems do not guarantee when a process will be given CPU time,
  so the time between packet arrival and algorithm execution varies.
- USB and network transmission: packet delivery times are not perfectly regular.
- Stimulator hardware: some devices have variable internal processing times between receiving a trigger command and delivering the stimulus.

In practice, both delay and jitter exist simultaneously in every part of the pipeline. Some amplifiers and stimulators are
better characterised than others, and it is worth measuring these quantities empirically for your specific hardware combination
before designing a phase estimation experiment.

Practically, to account for delay, you should measure the average and standard deviation of time from requesting a packet to receiving that packet on many trials, and perform a similar analysis for sending the stimulus.
You should account for this delay in your phase estimation algorithm by either detecting an earlier phase or subtracting this amount from the time you wait to schedule a trigger for ETP-like algorithms.

You want to minimize things that cause jitter: close background applications, utilize a `custom Python sleep function <https://stackoverflow.com/a/47735147/10213537>`_ not ``time.sleep`` as it introduces ~10ms of jitter and utilize lower latency methods for communicating with the stimulation hardware (e.g. for TMS, utilize a parallel port over a serial port).

Packet send rates
------------------

A challenge that receives less attention than delay and jitter, but that has equally important consequences for algorithm selection,
is the **packet send rate** of the amplifier.

EEG amplifiers do not send one sample at a time. They accumulate samples into packets and transmit them at a fixed rate.
The packet send rate determines how many times per second your software receives new data and can therefore run the phase
estimation algorithm. Common configurations range from a few packets per second up to several hundred.

For algorithms that **estimate the current phase**, such as PHASTIMATE, the algorithm fires when the estimated phase is
sufficiently close to the target phase. The number of opportunities to hit the target each second is bounded by the packet
send rate. Because EEG phase changes continuously and rapidly (a 10 Hz oscillation completes a full 360° cycle in 100 ms),
a low send rate means the algorithm checks phase only a handful of times per cycle. The probability of the estimated phase
being within a narrow tolerance of the target on any given check is low, so many cycles will pass without a trigger being
generated. In the extreme case, with very low send rates, this type of algorithm becomes practically unusable — you may
wait many seconds between successful detections.

For algorithms that **predict the timing of a future phase** — such as ETP — this constraint is much less severe. Rather
than waiting until the phase is "already there," the algorithm schedules a trigger to fire at the predicted future time.
As long as at least one prediction is made somewhere in the relevant window, a trigger can be scheduled accurately. The
packet send rate still affects the resolution of that prediction, but the algorithm is not fundamentally dependent on
happening to check at exactly the right moment.

This makes timing-prediction algorithms the more practical choice when working with amplifiers that have low or variable
packet send rates, or when communicating with an amplifier over a protocol that does not support high-frequency polling.
