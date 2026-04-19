What is EEG phase estimation?
==============================

EEG phase triggered stimulation (e.g. through tES, TMS or visual/auditory stimulation) has gained a great deal of attention over the past several years. However, the number of studies testing phase triggered stimulation paradigms are limited because of the technical challenge associated with delivering stimulation at different EEG phases. This guide talks about what the specific technical challenges are and how EEGPhasePy helps overcome that challenge.


Edge effects
-------------

Currently, all methods of EEG phase estimation require at least one convolution operation: filtering. Convolutions naturally produce distortions near the start and end of signals because of the limited data at each end. This in turn distorts the phase information at each end making it challenging to simply filter then extract phase through a Hilbert transform or similar method.

Phase estimation
--------------------------

All phase estimation models are inherently trying to overcome the edge effect problem of one or more convolutions (e.g. involved in filtering the signal). In solving this problem, there are two major approaches: 1) estimating the current phase or 2) predicting the timing of the target phase. 

Estimating the current phase, just involves predicting what the phase at t = 0 for the current window of data is. To deliver a stimulus using these algorithms, you would just need to trigger the stimulus when the estimated phase is "close enough" to the target phase. Predicting the timing of the future phase, just involves predicting the time from the current window that the target phase would occur at. With these algorithms, you would have to "schedule" a pulse, which is essentially just waiting for the time the next phase would occur at then triggering the stimulus.

How EEGPhasePy helps
---------------------

EEGPhasePy includes implementation of both types of phase estimation algorithms. It also includes implementations of calculations of different stats for quantifying accuracy or reliability of EEG phase estimation + some visualization helpers.
