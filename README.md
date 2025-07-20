# EEGPhasePy
EEGPhasePy is an open-source toolkit for real-time phase estimation from electroencephlography (EEG). It is currently in a work-in-progress state. 
For a brief intro to the potential applications of EEG phase estimation see https://pmc.ncbi.nlm.nih.gov/articles/PMC10881194/

## Plan
The goal is to replicate the mainstream EEG phase estimation algorithms including: autoregressive (AR) <a href="https://pubmed.ncbi.nlm.nih.gov/29191438/">(Zrenner et al., 2018)</a> and educated temporal prediction (ETP) <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8293904/">(Shrinpour et al., 2020)</a>. Along with several helper methods for offline analysis of phase estimation experiments, producing figures such as polar histograms or average +- std waveforms. The goal with this package as well is to include methods for overcoming challenges of applying phase estimation in real-time such as a jitter free timing function and auto-incoorporation of delay into all supported algorithms.
