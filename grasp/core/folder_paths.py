"""Filesystem layout for GRASP data caches.

Authors
-------
- Pietro Ferraiuolo : written in 2024

Notes
-----
Path *constants* are defined eagerly at import time so downstream code can
refer to them, but **no directories are created at import time**. Call
:func:`initialize_data_layout` explicitly (e.g. from the future ``grasp.cli``
entry point or from a first cache write) or use :func:`ensure_data_dir` to
create individual subdirectories on demand.
"""

import logging as _logging
import os as _os

_logger = _logging.getLogger("grasp.core.folder_paths")

BASE_PATH = _os.path.dirname(_os.path.dirname(__file__))
try:
    BASE_DATA_PATH = _os.path.join(_os.environ["GRASPDATA"])
except KeyError:
    _logger.info(
        "No GRASPDATA environment variable found. Using the HOME folder."
    )
    BASE_DATA_PATH = _os.path.join(_os.path.expanduser("~"), "graspdata")

SYS_DATA_FOLDER = _os.path.join(BASE_PATH, "sysdata")
CATALOG_FILE = _os.path.join(BASE_PATH, "sysdata", "_Catalogue.xlsx")
KING_INTEGRATOR_FOLDER = _os.path.join(BASE_PATH, "analyzers", "_king")
MCLUSTER_SOURCE_CODE = _os.path.join(BASE_PATH, "analyzers", "_mcluster")
R_SOURCE_FOLDER = _os.path.join(BASE_PATH, "analyzers", "_Rcode")
FORMULARY_BASE_FILE = _os.path.join(SYS_DATA_FOLDER, "base.frm")
QUERY_DATA_FOLDER = _os.path.join(BASE_DATA_PATH, "query")
KING_MODELS_FOLDER = _os.path.join(BASE_DATA_PATH, "models")
SIMULATION_FOLDER = _os.path.join(BASE_DATA_PATH, "simulations")
UNTRACKED_DATA_FOLDER = _os.path.join(QUERY_DATA_FOLDER, "UntrackedData")

_DATA_DIRS = (
    BASE_DATA_PATH,
    QUERY_DATA_FOLDER,
    KING_MODELS_FOLDER,
    SIMULATION_FOLDER,
    UNTRACKED_DATA_FOLDER,
)


def initialize_data_layout(*, verbose: bool = True) -> tuple[str, ...]:
    """Create the GRASP data-cache directory tree on disk.

    Idempotent. Logs at ``INFO`` for each newly created directory.

    Parameters
    ----------
    verbose : bool, optional
        If ``False`` the function does not log per-directory creation,
        useful for very chatty CI environments.

    Returns
    -------
    tuple[str, ...]
        The directories that were created during this invocation (existing
        ones are skipped silently).
    """
    created: list[str] = []
    for path in _DATA_DIRS:
        if not _os.path.exists(path):
            _os.makedirs(path, exist_ok=True)
            created.append(path)
            if verbose:
                _logger.info("Created GRASP data directory: %s", path)
    return tuple(created)


def ensure_data_dir(path: str) -> str:
    """Create ``path`` (and any missing parent) if it does not yet exist.

    Used by query/save routines so that the directory layout is constructed
    lazily on first write rather than as a side effect of ``import grasp``.

    Parameters
    ----------
    path : str
        Absolute or relative directory path to create.

    Returns
    -------
    str
        The (now-existing) directory path.
    """
    if not _os.path.exists(path):
        _os.makedirs(path, exist_ok=True)
        _logger.info("Created GRASP data directory: %s", path)
    return path


def CLUSTER_DATA_FOLDER(name: str) -> str:
    """Return the cluster's data path (does not create it)."""
    return _os.path.join(QUERY_DATA_FOLDER, name.upper())


def CLUSTER_MODEL_FOLDER(name: str) -> str:
    """Return the cluster's model path (does not create it)."""
    return _os.path.join(KING_MODELS_FOLDER, name.upper())
