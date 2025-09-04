Optimizing PHASTIMATE parameters
=========================================================================

The PHASTIMATE parameters of `window_edge` and `ar_order` significantly impact the performance of PHASTIMATE performance. In certain 
cases, such as when dealing with high inter-individual variability or working with a frequency band that is not commonly studied, using 
an optimization algorithm to select these parameters is ideal. We have implemented 2 approaches for optimizing these parameters.

Genetic Optimization
---------------------

.. nbgallery::
   ../examples/ar-genetic-optimization

Bayesian Optiization 
---------------------
.. nbgallery::
   ../examples/ar-bayesian-optimization