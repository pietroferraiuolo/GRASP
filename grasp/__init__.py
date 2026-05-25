"""
GRASP - Globular clusteR Astrometry and Photometry Software
===========================================================
Author(s)
---------
- Pietro Ferraiuolo : Written in 2024

Description
-----------
GRASP is a tool for analysing the data of globular clusters, with major
attention to those found in the Gaia archive. While born for GCs and Gaia
data, it has been kept general enough to be applied to other datasets and
problems, with an ample set of calculus and analysing tools.

The software exposes the following sub-modules, each tackling a specific task:

- ``analyzers``: data analysis utilities. Wraps the Fortran King integrator
  and the McLuster simulator, and historically hosted a set of R scripts for
  statistical fitting (now superseded by Python implementations).
- ``stats``: statistical analysis (regression, GMM, bootstrap, ...).
- ``plots``: convenience plotting routines for astrometric/photometric data.
- ``functions``: analytical helpers (currently ``CartesianConversion``).
- ``gaia``: Gaia archive query interface.
- ``formulary``: symbolic formula management and numerical evaluation.
- ``_utility``: ``Cluster`` and ``Sample`` data classes.
- ``core``: filesystem and IO helpers.
"""

from .__version__ import (
    __title__,
    __long_title__,
    __description__,
    __version__,
    __author__,
    __author_email__,
    __license__,
    __url__,
)

from .gaia._zero_point import zero_point_correction
from .gaia.query import GaiaQuery, available_tables
from .analyzers.mcluster import mcluster_run, docs as mcluster_docs
from .analyzers import calculus
from .analyzers._Rcode.r2py_models import (
    RegressionModel,
    GaussianMixtureModel,
)
from ._utility.base_classes import BaseFormula
from ._utility.cluster import Cluster, available_clusters
from ._utility.sample import Sample, GcSample
from .formulary import Formulary, load_base_formulary

from . import stats
from . import plots
from . import analyzers
from . import core

osu = core.osutils
load_data = osu.load_data
gpaths = core.folder_paths


def dr3():
    """Instance a GaiaQuery with DR3"""

    print(
        """
            ..............
         ..:;;..:;;;;;:::::;;
       ;;;;;;::.::;;;;;;;;;;;;;
      ;;;;.:;;;..;XXXXXX.::....:           GAIA QUERY MODULE
     :;::;::..+XXXXXXXXX+:;;;;;;:
    ;::;:.:;;:XXXXXXXXXXX::::::::;        __ _  __ _(_) __ _
    .;;..;;;:.:XXXXXXXXX$$$$$$$$$$X.     / _` |/ _` | |/ _` |
    :;..:;;;..:xXXXXXXX$$$$$$$$$$$$X    | (_| | (_| | | (_| |
    :;:.:;;.XXXXXXXXX$$$$$$$$$$$$$$;     \\__, |\\__,_|_|\\__,_|
    .;;:.:X$$$$$$$$$$$$$$$$$$$$$$X.      |___/
    ..:;:$$$$$$$$$$$$$$$$$$$$$$X;.
     :;;;$$$$$$$$$$$$$$$$$$$$::;;             INITIALIZED
      ...;$$$$$$$$$$$$$$x;:;;;;;
        ......:;:....;;;;;;;:.
          ::::::::::;;;::...
            ....::::.....
"""
    )
    return GaiaQuery()


__all__ = [
    "__title__",
    "__long_title__",
    "__description__",
    "__version__",
    "__author__",
    "__author_email__",
    "__license__",
    "__url__",
    "available_clusters",
    "available_tables",
    "zero_point_correction",
    "GaiaQuery",
    "mcluster_run",
    "mcluster_docs",
    "calculus",
    "RegressionModel",
    "GaussianMixtureModel",
    "BaseFormula",
    "Cluster",
    "Formulary",
    "load_base_formulary",
    "Sample",
    "GcSample",
    "stats",
    "plots",
    "analyzers",
    "core",
    "load_data",
    "osu",
    "gpaths",
    "dr3",
]
