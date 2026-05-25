"""Sphinx configuration for GRASP.

The configuration is intentionally minimal. ``sphinx-build`` is expected
to be run from the repository root or from ``docs/`` so that
``grasp`` is importable. The version is pulled from the installed
package metadata.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

sys.path.insert(0, os.path.abspath("../.."))

project = "GRASP: Globular clusteR Astrometry and Photometry Software"
copyright = "2024-2026, Pietro Ferraiuolo"
author = "Pietro Ferraiuolo"

try:
    release = _pkg_version("grasp")
except PackageNotFoundError:
    from grasp.__version__ import __version__ as release  # type: ignore[no-redef]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

autodoc_mock_imports = [
    "setup",
    "rpy2",
    "astroML",
    "astroquery",
]

numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "sympy": ("https://docs.sympy.org/latest", None),
}

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
