# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../../"))

project = 'EEGPhasePy'
copyright = '2025, EEGPhasePy Authors'
author = 'EEGPhasePy Authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'nbsphinx'
]

# autosummary_generate = True
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []

bibtex_bibfiles = ['references.bib']
bibtext_default_style = 'unsrt'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
