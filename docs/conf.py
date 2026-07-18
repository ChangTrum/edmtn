"""Sphinx configuration for the edmtn documentation.

The ``edmtn`` package is made importable by the documentation environment
itself (a ``.pth`` file pointing at the repository ``src/`` layout), so this
file performs no ``sys.path`` manipulation.
"""

from __future__ import annotations

import edmtn

# -- Project ---------------------------------------------------------------

project = "edmtn"
release = edmtn.__version__
version = release
author = "ChangTrum"
copyright = "2026, ChangTrum"

# -- General ---------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

exclude_patterns = [
    "_build",
    # Context-only material (the arXiv preprint PDF and the private technical
    # plan); not part of the rendered documentation.
    "reference/**",
]

# -- MyST (Markdown sources) -----------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
myst_heading_anchors = 3

# -- autodoc / napoleon ----------------------------------------------------

# The public API of each layer package is the set of names it exports via
# ``__all__``; ``automodule`` documents exactly that list.
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True

# -- HTML output -----------------------------------------------------------

html_theme = "furo"
html_title = "edmtn"
