"""
GAIA - GLOBULAR CLUSTER ANALYSIS SOFTWARE
=========================================
Author(s)
---------
- Pietro Ferraiuolo : Written in 2024

Description
-----------
This software is designed to analyze the data of globular clusters, with major
attention to those found in the Gaia database. While born as a tool for GCs and
Gaia data, effort has been made to make it more general and applicable to other
datasets and problems, whit an ample set of calculus and analyzing tools.
The software includes the subsequent series of submodule, each tackling a
specific task:

analyzers: 
    Module for the analysis of the data. This module also contains the source code
    for the Fortran King integration and the McLuster software, wrapped with
    Python into their own modules. Also contains the various R source codes
    for the statistical analysis.
statistics: 
    Module for the statistical analysis of data, including bayesian statistic
    methods with machine learning techniques.
plots: 
    Module with inclues a series of usefull plotting functions, for better 
    visualizing statistical data, as well as visualization of cluster's data.
functions: 
    Module with a series of analythical functions for the computation
    of various cluster and source's parameters. (eg. distance, mean motion, ecc...)
_query: 
    To be changed
_cluster: 
    Module for the cluster class, which contains all the information of a specified cluster.
_utility: 
    Module with a series of utility functions, used in the various modules.
    (eg. load_data, get_file_list, tnlist)

"""
from .__version__ import *
from grasp.core.osutils import load_data, load_simulation_data
import grasp.core.folder_paths as gpaths

from grasp.gaia._zero_point import zero_point_correction
from grasp.gaia.query import GaiaQuery, available_tables
from grasp.analyzers.mcluster import mcluster_run, docs as mcluster_docs
from grasp.analyzers import calculus
from grasp.analyzers._Rcode.r2py_models import RegressionModel, GMModel, PyRegressionModel
from grasp._utility.base_formula import BaseFormula
from grasp._utility.cluster import Cluster, available_clusters
from grasp._utility.sample import Sample

from grasp import stats
from grasp import plots
from grasp.formulary import Formulary, load_base_formulary


def dr3():
    """Instance a GaiaQuery with DR3"""

    print("""
GAIA - GLOBULAR CLUSTERS ANALYSIS SOFTWARE
            ..............
         ..:;;..:;;;;;:::::;;
       ;;;;;;::.::;;;;;;;;;;;;;
      ;;;;.:;;;..;XXXXXX.::....:
     :;::;::..+XXXXXXXXX+:;;;;;;:
    ;::;:.:;;:XXXXXXXXXXX::::::::;        __ _  __ _(_) __ _
    .;;..;;;:.:XXXXXXXXX$$$$$$$$$$X.     / _` |/ _` | |/ _` |
    :;..:;;;..:xXXXXXXX$$$$$$$$$$$$X    | (_| | (_| | | (_| |
    :;:.:;;.XXXXXXXXX$$$$$$$$$$$$$$;     \__, |\__,_|_|\__,_|
    .;;:.:X$$$$$$$$$$$$$$$$$$$$$$X.      |___/
    ..:;:$$$$$$$$$$$$$$$$$$$$$$X;.
     :;;;$$$$$$$$$$$$$$$$$$$$::;;
      ...;$$$$$$$$$$$$$$x;:;;;;;
        ......:;:....;;;;;;;:.
         ::::::::::;;;::...

             INITIALIZED

""")
    return GaiaQuery()


__all__ = [
    'zero_point_correction',
    'GaiaQuery',
    'mcluster_run',
    'calculus',
    'RegressionModel',
    'GMModel',
    'BaseFormula',
    'Cluster',
    'Formulary',
    'load_base_formulary',
    'Sample',
    'stats',
    'plots',
    'dr3'
]