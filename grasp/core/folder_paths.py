"""
Author(s)
---------
    - Pietro Ferraiuolo : written in 2024

Description
-----------
"""

import os as _os

BASE_PATH = _os.path.dirname(_os.path.dirname(__file__))
try:
    BASE_DATA_PATH      = _os.path.join(_os.environ['GRASPDATA'])
except KeyError:
    print("No GRASPDATA environment variable found. Using the HOME folder.")
    BASE_DATA_PATH      = _os.path.join(_os.environ['HOME'], 'graspdata')
SYS_DATA_FOLDER         = _os.path.join(BASE_PATH, "sysdata")
CATALOG_FILE            = _os.path.join(BASE_PATH, 'sysdata', "_Catalogue.xlsx")
KING_INTEGRATOR_FOLDER  = _os.path.join(BASE_PATH, "analyzers", "_king")
MCLUSTER_SOURCE_CODE    = _os.path.join(BASE_PATH, "analyzers", "_mcluster")
R_SOURCE_FOLDER         = _os.path.join(BASE_PATH, "analyzers", "_Rcode")
FORMULARY_BASE_FILE     = _os.path.join(SYS_DATA_FOLDER, "base.frm")
QUERY_DATA_FOLDER       = _os.path.join(BASE_DATA_PATH, "query")
KING_MODELS_FOLDER      = _os.path.join(BASE_DATA_PATH, "models")
SIMULATION_FOLDER       = _os.path.join(BASE_DATA_PATH, "simulations")
UNTRACKED_DATA_FOLDER   = _os.path.join(QUERY_DATA_FOLDER, "UntrackedData")

paths = [
    BASE_DATA_PATH,
    QUERY_DATA_FOLDER,
    KING_MODELS_FOLDER,
    SIMULATION_FOLDER,
    UNTRACKED_DATA_FOLDER,
]
for p in paths:
    if not _os.path.exists(p):
        _os.mkdir(p)


def CLUSTER_DATA_FOLDER(name: str):
    """
    Returns the cluster's data path.
    """
    path = _os.path.join(QUERY_DATA_FOLDER, name.upper())
    return path


def CLUSTER_MODEL_FOLDER(name: str):
    """
    Returns the cluster's model path.
    """
    path = _os.path.join(KING_MODELS_FOLDER, name.upper())
    return path
