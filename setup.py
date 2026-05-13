"""Shim retained for ``pip install -e .`` callers on older toolchains.

All real packaging configuration lives in ``pyproject.toml`` (PEP 621).
"""

from setuptools import setup

setup()
