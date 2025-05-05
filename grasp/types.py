from typing import (
    TYPE_CHECKING,
    Callable,
    TypeVar,
    Union,
    Any,
    Optional,
    TypeAlias
)
from numpy.typing import ArrayLike
from pandas import DataFrame
from astropy.table import Table, QTable
import sympy as _sp

if TYPE_CHECKING:
    from grasp._utility.sample import Sample
    from grasp._utility.cluster import Cluster
    from grasp.analyzers._Rcode.r2py_models import (
        GaussianMixtureModel,
        PyRegressionModel,
        RegressionModel
    )


TabularData: TypeAlias = Union[
    "Sample",
    DataFrame,
    Table,
    QTable
]

AstroTable: TypeAlias = Union[
    DataFrame,
    Table,
    QTable
]

GcInstance: TypeAlias = Union[
    str,
    "Cluster",
]

FittingFunc: TypeAlias = Union[str, Callable[..., float]]

RegressionModels: TypeAlias = Union[
    "RegressionModel",
    "PyRegressionModel"
]

RRegressionModel: TypeAlias = "RegressionModel"
PRegressionModel: TypeAlias = "PyRegressionModel"

GMModel: TypeAlias = "GaussianMixtureModel"

Array: TypeAlias = Union[
    ArrayLike,
    list[int],
    list[float],
    list[complex],
    list[int,float,complex]
]

AnalyticalFunc : TypeAlias = Union[
    _sp.Basic,
    _sp.Add,
    _sp.Mul,
    _sp.Pow,
    _sp.Function,
    _sp.Equality
]
Variables: TypeAlias = Union[_sp.Symbol,_sp.NumberSymbol]
