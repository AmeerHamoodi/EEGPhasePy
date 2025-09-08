.. EEGPhasePy documentation master file, created by
   sphinx-quickstart on Fri Aug 29 15:26:58 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EEGPhasePy documentation
========================

Tookit for developing and analyzing real-time and pseudo-real-time EEG phase estimation.

Getting started
----------------
To get started with EEGPhasePy, you will first need to install the packaeg from PyPi by running the command:

.. code-block:: Python
    :caption: Install EEGPhasePy
    
    pip install EEGPhasePy

Next, look through our usage guide for phase estimation. We recommend beginning with ETP phase estimation first.


Usage
----------------
.. toctree::
   :maxdepth: 3

   usage/etp
   usage/ar
   usage/optimization
   usage/plots
   usage/stats


Examples
----------

.. nbgallery::
   examples/etp-based-phase-estimation
   examples/ar-phase-estimation
   examples/ar-genetic-optimization
   examples/ar-bayesian-optimization

Reference
----------------
.. toctree::
   :maxdepth: 3

   api