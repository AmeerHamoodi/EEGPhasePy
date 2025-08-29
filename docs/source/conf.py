# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EEGPhasePy'
copyright = '2025, EEGPhasePy Authors'
author = 'EEGPhasePy Authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.autosummary', 
    'sphinx.ext.napoleon',
]

autosummary_generate = True
autodoc_typehints = "description"

# Napoleon settings
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_keyword = True
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = []

apidoc_modules = [
    {'path': '../../EEGPhasePy/', 'destination': './api-reference'}
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
