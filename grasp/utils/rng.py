"""Reproducible random number generation helpers (P3-4).

GRASP previously seeded :class:`astroML.density_estimation.XDGMM` with
``random_state=int(time.time())``, making fits irreproducible and
data-dependent in subtle ways. This module wraps
:func:`numpy.random.default_rng` so every public stochastic API in
GRASP can accept a deterministic ``seed`` keyword.

Typical usage::

    from grasp.utils.rng import default_rng

    rng = default_rng(20260513)
    sample = rng.normal(size=1000)

Passing ``None`` (the default) yields a fresh, non-deterministic generator,
matching the behaviour of :func:`numpy.random.default_rng`.
"""

from __future__ import annotations

from typing import Union

import numpy as np

SeedLike = Union[None, int, "np.random.SeedSequence", "np.random.Generator"]


def default_rng(seed: SeedLike = None) -> np.random.Generator:
    """Return a NumPy :class:`~numpy.random.Generator`.

    Parameters
    ----------
    seed : int, :class:`~numpy.random.SeedSequence`, :class:`~numpy.random.Generator` or None
        Seed forwarded to :func:`numpy.random.default_rng`. If already a
        ``Generator`` instance it is returned unchanged so users can
        thread a single RNG through a chain of operations.

    Returns
    -------
    rng : numpy.random.Generator
        A deterministic generator when ``seed`` is provided, otherwise
        a fresh non-deterministic one seeded from the OS.

    Notes
    -----
    Always prefer this helper over :class:`numpy.random.RandomState` and
    over module-level ``numpy.random.seed`` -- those carry global state
    that breaks parallelism and reproducibility guarantees.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
