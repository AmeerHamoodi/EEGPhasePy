---
title: "EEGPhasePy: A toolbox for real-time EEG phase estimation"
tags:
  - Python
  - electroencephalography
  - EEG-TMS
authors:
  - name: Ameer Hamoodi
    orcid: 0009-0006-4375-8104
    corresponding: true
    affiliation: "1, 2"
  - name: Christian Brodbeck
    orcid: 0000-0001-8380-639X
    affiliation: 1
  - name: Mustaali Hussain
    orcid: 0009-0004-7096-2763
    affiliation: 3
  - name: Aimee Nelson
    orcid: 0000-0003-1279-0815
    affiliation: "1, 3"
affiliations:
  - name: School of Biomedical Engineering, McMaster University, Hamilton, ON, Canada
    index: 1
  - name: Department of Medicine, McMaster University, Hamilton, ON, Canada
    index: 2
  - name: Department of Kinesiology, McMaster University, Hamilton ON, Canada
    index: 3
date: 9 July 2026
bibliography: paper.bib
---

# Summary

Electroencephalography (EEG) phase triggered transcranial magnetic stimulation (TMS) has
risen in popularity for both understanding basic neurophysiology and as a
novel potential therepeutic avenue for various neurological and psychiatric
illnesses [@ZRENNER2024545]. However, targeting specific EEG phases for TMS delivery is technically
challenging limiting its use by labs interested in investigating EEG phase-triggered TMS [@McIntosh_2020].
`EEGPhasePy` is an open-source toolbox for both real-time implementation of EEG phase
estimation algorithms and offline analysis. We have included comprehensive documentation alongside
multiple tutorials and a practical walkthrough of EEG phase estimation in real-time.

# Statement of need

Closed-loop non-invasive neurostimulation has risen in popularity as a potential
novel therepeutic avenue for many neurological conditions including depression,
Alzheimer's disease and chronic pain [@ZRENNER2024545]. A common EEG target that holds potential for
studying the physiological basis of EEG as well as in clinical applications is EEG phase
in the alpha and theta bands. Delivering TMS at specific phases in alpha and theta appears to result in
higher/lower corticospinal and cortical excitability [@Zrenner2018-hh; @Gordon2022]. A small number of studies have
been conducted to test the clinical potential of repetitive TMS (rTMS) at triggered at specific
EEG phases, with positive findings in depression and post-stroke rehabilitation [@George2023; @Wala2024]. A major challenge
for neurophysiologists who want to apply phase-triggered rTMS to their work is the technical difficulty
of accurately delivering rTMS at specific phases and measuring the accuracy of their approaches. This was
an issue for our lab when first attempting to conduct phase-triggered TMS experiments.

`EEGphasePy` was developed to be used by neurophysiology researchers
to conduct phase-triggered TMS experiments. The package's documentation was shaped by
questions colleagues unfamiliar but interested in phase estimation asked. It contains guides
for overcoming common technical challenges such as jitter in TMS trigger timing, EEG packet delay
and EEG packet send rate variations across systems.

# State of the field

At present, the field consists of open-source implementations of particular phase estimation algorithms.
Additionally, the `PHASTIAMTE` package was released by Zrenner et al. in 2020 [@Zrenner2020-zb] but only implements
autoregressive (AR) phase estimation and genetic optimization for AR phase estimation parameters in MATLAB. It was also not
designed for external contributions. Further, multiple phase estimation algorithms
lack open-source implementations [@Liu2025].

**Build vs contribute:** We opted to develop a new library rather than contribute to MNE-Python because MNE was not designed
for use in real-time applications. The `Raw` class is not amenable to sliding window creation, which is necessary for real-time
EEG processing. The MNE-Python real-time package was discontinued in favor of MNE-LSL and MNE-LSL relies on Lab Streaming Layer (LSL) for communication. However, multiple EEG amplifiers, including our amplifiers in lab, do not use LSL for streaming data. By creating a separate library, we ensure phase estimation can be used across various hardware set ups.

# Software design

`EEGPhasePy` was designed with two goals in mind: 1) ensure the package is easily used by neurophysiologists
without significant coding experience and 2) ensure new algorithms and metrics can easily be added to the package. The first
goal was accomplished by including non-software related documentation such as our phase estimation guide and
guide for real-time phase estimation. These guides arose from discussions with non-engineering colleagues. Additionally,
we followed an `sklearn`-like architecture for designing phase estimation algorithms. Each phase estimation algorithm
is a child of a parent `Estimator` containing a `fit`, `predict` and various metric/visualization methods. The goal of
our architecture is to make using an estimation algorithm intuitive so that after learning to implement one algorithm,
the user could theoretically easily implement other algorithms. Additionally, as the use of large-language models (LLM)
has become more prevalent, we aimed for the choice of wording of `EEGPhasePy`'s methods to be representative of the underlying
function of those methods. This was done to ensure that scientists could easily understand what the code generated by LLMs is doing.

For goal 2, we designed a general `Estimator` class that contains both a `fit` and `predict` method. This approach makes it easy to
train the phase estimator on resting-state EEG and then use it new recordings by pickeling the trained model. Further, the `Estimator` class
contains common metrics and visualizations that allow scientists to test the accuracy of their phase estimation algorithm. As new metrics become
adopted by the literature, they can be easily added to the `EEGPhasePy` library and become widely accessible as a result of the `Estimator` class.
We implemented automated tests for each phase estimation algorithm on synthetic EEG data, this ensures the algorithm is performing
with expected accuracy for that model. Additionally, we created GitHub actions to automatically test for linting errors using `flake8`, run
the tests using `PyTest` and build the docs onto `readthedocs`. Our GitHub actions run these tests across multiple Python versions for
compatibility.

# Research impact statement

While `EEGPhasePy` is still in its early stages, we have utilized it to benchmark alpha and theta phase estimation
in dementia patients . We are also currently using it in ongoing studies in our lab. We do anticipate significant
future use as multiple labs have indicated interest in performing phase-triggered TMS experiments in both clinical and
non-clinical populations, but have been unable to do so due to the technical challenge of performing EEG phase estimation.
Additionally, multiple papers have cited avoiding phase triggered TMS and opting for alternative modes of state-dependent
TMS due to technical challenges (). We anticipate that this package will have significant future impact on our field as
with the rise of LLMs, a validated package of phase estimation algorithms and guides for implementing phase estimation in real-time
will make it substantially easier and more trustworthy for LLMs to aid scientists in the implementation of EEG phase estimation
for various experiments. Further, due to the extensibility of our package, new algorithms, metrics and visualization modes can be
easily integrated.

# AI usage disclosure

Generative AI was used as an additional code reviewer (Claude code and GitHub CoPilot) and did not replace human review of the code or
human testing of the code. Generative AI made minor contributions to code refactoring and bug fixes (Claude code) as well as in formatting
documentation. All documentation and code was reviewed thoroughly by the authors.

# Acknowledgements

AH gratefully acknowledge the contributions of the Alzheimer Society Research Program, made possible by the Alzheimer Society of Canada. Their support was essential in making this research possible.

# References
