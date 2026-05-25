"""Pluggable fit backends for :mod:`grasp.stats`.

GRASP historically delegated regression and Gaussian-mixture fits to R via
:mod:`rpy2`. Phase 3 of the cleanup introduces a pure-Python alternative,
selectable through a ``backend`` keyword argument or the
``GRASP_R_BACKEND`` environment variable. The Python implementation is now
the **default**; the R path is kept available for parity testing and emits
a :class:`DeprecationWarning` every time it is invoked.

Sub-modules
-----------
- :mod:`._python` -- :mod:`lmfit` / :mod:`scikit-learn` / :mod:`statsmodels`
  ports. Recommended.
- :mod:`._r` -- thin wrappers around the existing R routines.
"""

from . import _python  # noqa: F401

__all__ = ["_python"]
