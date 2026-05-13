"""Small utility helpers shared across :mod:`grasp`.

Currently exposes :mod:`grasp.utils.rng` for deterministic random number
generation.
"""

from .rng import default_rng

__all__ = ["default_rng"]
