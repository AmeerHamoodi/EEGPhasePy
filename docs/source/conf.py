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
copyright = '2026, EEGPhasePy Authors'
author = 'EEGPhasePy Authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx_gallery.gen_gallery'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None)
}

sphinx_gallery_conf = {
    'examples_dirs': '../examples',   # path to your example scripts
    'gallery_dirs': 'gallery_examples',  # path to where to save gallery
    'ignore_pattern': r"(^|/)(GALLERY_HEADER\.rst|__init__\.py)$",
}

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
