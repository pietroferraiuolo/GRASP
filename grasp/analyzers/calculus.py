"""
Author(s):
----------
- Pietro Ferraiuolo : written in 2024

Copyright : Pietro Ferraiuolo (pietro.ferraiuolo@inaf.it)

Description
-----------

How to Use it
-------------
"""

import sympy as _sp
import numpy as _np
from . import _glpoints
import multiprocessing as _mp
from grasp import types as _t


def compute_numerical_function(
    func: _t.AnaliticalFunc, variables: _t.Variables, var_data: list[_t.Array]
) -> _t.Array:
    """

    Compute the numerical value of a function, passing by the function, it's
    variables and the data associated to the variables.

    Parameters
    ----------
    func : sympy.core.function
        Function to compute. Must be a sympy expression.
    variables : list of sympy variables
        Variables of the function, as sympy symbols, organizaed in a list.
    var_data : list of ndarray
        Numerical values of the variables, organized in a list ordered the same
        way as the order of the variable list.

    Returns
    -------
    computed_func : float | ArrayLike
        List of values of the function computed for each data point.

    Notes
    -----
    When the computation exceeds 30 seconds running on all cores, which should
    happen with ~2000 points per core, the expression gets compiled with numpy,
    significantly increasing computations times, while being compatible with the
    full computation precision up to 10e-9.
    """
    n_cores = _mp.cpu_count()
    N = _np.array(var_data).shape[-1]
    Nf = len(var_data)
    if N < (2000 * n_cores) and Nf < 3:
        # Multicore computation
        print(f"Computation using all {n_cores} cores.")
        val_dicts = _data_dict_creation(variables, var_data)
        computed_func = _multicore_computation(n_cores, func, val_dicts)
    else:
        # lambdify computation
        print(
            "WARNING: computation time exceeding 30s. Compiling expression with NumPy."
        )
        computed_func = _lambdified_computation(func, variables, var_data)
    return _np.array(computed_func)


def compute_error(
    err_func: _t.AnaliticalFunc,
    variables: _t.Variables,
    var_data: list[_t.Array],
    var_errors: list[_t.Array],
    corr_values: _t.Optional[list[_t.Array]] = None,
) -> _t.Array:
    """
    Numerical computation of the error-formula for the input function.

    Parameters
    ----------
    err_func : sympy function
        The symbolic function for which the error needs to be computed.
    variables : list of sympy symbols
        The list of symbolic variables in the function, both the data and the
        data error variables.
    var_data : list of ndarray
        The list of data arrays corresponding to each variable.
    var_errors : list of ndarray
        The list of error arrays corresponding to each variable.
    corr_values : list, optional
        The list of correlation values between the variables. The default is None.

    Returns
    -------
    computed_error : list of floats
        The computed numerical error for the input function.

    """
    data = var_data + var_errors
    if corr_values is not None:
        data += corr_values
    computed_error = compute_numerical_function(err_func, variables, data)
    return computed_error


def error_propagation(
    func: _t.AnaliticalFunc, variables: _t.Variables, correlation: bool = False
) -> dict[str, _t.Union[_t.AnalyticalFunc, _t.Variables]]:
    """
    Computes the imput function's error with the standard error propagation
    formula:
    :math:`\\varepsilon(F)=\sqrt{\sum_{i=1}^n {\\bigg(\\frac{\partial F}{\partial x_i}\\bigg)^2\\varepsilon_{x_i}^2} + 2\sum_{i\\ne j}{\\bigg(\\frac{\partial F}{\partial x_i}\\bigg)\\bigg(\\frac{\partial F}{\partial x_j}\\bigg)\\varepsilon_{x_i}\\varepsilon_{x_j}}\\rho(x_i,x_j)}`

    Parameters
    ----------
    func : sympy expression
        DESCRIPTION.
    variables : list of sympy symbols
        DESCRIPTION.
    corr : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    result : dict
        A dictionary containing the results.
        Keys:
        - "error_formula" : sympy expression
        - "error_variables" : list of sympy symbols
        - "correlations" : list of sympy symbols (only if corr=True)
    """
    if correlation:
        corr = _np.ones((len(variables), len(variables)), dtype=object) - _np.eye(
            len(variables), dtype=object
        )
    else:
        corr = _np.eye(len(variables), dtype=object)
    errors = []
    for i, var1 in enumerate(variables):
        errors.append(_sp.symbols("epsilon_{}".format(var1)))
        for j, var2 in enumerate(variables):
            if i != j:
                corr[i][j] = corr[i][j] * _sp.symbols(f"rho_{var1}_{var2}")
    # Partial derivatives computation
    partials = [_sp.diff(func, var) for var in variables]
    # Quadratic errors sum
    sum_of_squares = sum(
        (partials[i] ** 2 * errors[i] ** 2) for i, _ in enumerate(variables)
    )
    # Correlation sums
    sum_of_correlations = 0
    for i, _ in enumerate(variables):
        for j, _ in enumerate(variables):
            if i != j:
                sum_of_correlations += (
                    partials[i] * partials[j] * errors[i] * errors[j] * corr[i][j]
                )
    # Total error propagation
    error_formula = _sp.sqrt(sum_of_squares + sum_of_correlations)
    corrs = []
    for x in corr:
        for c in x:
            if c != 0:
                corrs.append(c)
    try:
        assert len(corrs) % 2 == 0, "Correlations must be in pairs"
        assert len(corrs) != 0, "No correlation, skipping"
        assert len(corrs) == len(variables) * (
            len(variables) - 1
        ), "Correlations not matching the number of variables"
        for i in range(0, len(corrs), 2):
            eformula = _sp.nsimplify(error_formula.subs(corrs[i + 1], corrs[i]))
            corrs.pop(i + 1)
    except AssertionError:
        eformula = error_formula
    erp = {
        "error_formula": eformula,
        "error_variables": {"variables": variables, "errors": errors},
    }
    if correlation:
        erp["error_variables"]["corrs"] = corrs
    return erp


def gaus_legendre_integrator(
    fnc: _t.Callable[..., float], a: float, b: float, points: int
) -> float:
    """
    Integrates the function fcn(x) between a and b using the Gauss-Legendre method.

    Parameters
    ----------
    fcn : callable
        The function to integrate, written as a callable python function.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    points : int
        The number of points to use for the integration (20, 40, 80, 96).

    Returns
    -------
    area : float
        The integral area of fcn(x) from a to b.

    Notes
    -----
    Note that the imput function must be a 'pythonic' function, and not a sympy
    functions, as it is not implemented yet.
    """
    if points == 20:
        x = _np.array(_glpoints.x20)
        w = _np.array(_glpoints.w20)
    elif points == 40:
        x = _np.array(_glpoints.x40)
        w = _np.array(_glpoints.w40)
    elif points == 80:
        x = _np.array(_glpoints.x80)
        w = _np.array(_glpoints.w80)
    elif points == 96:
        x = _np.array(_glpoints.x96)
        w = _np.array(_glpoints.w96)
    else:
        raise ValueError("Supported point values are 20, 40, 80, and 96.")
    area = 0.0
    try:
        for i in range(int(points / 2)):
            xi = (b - a) / 2.0 * x[i] + (b + a) / 2.0
            area += w[i] * fnc(xi)
            xi = -(b - a) / 2.0 * x[i] + (b + a) / 2.0
            area += w[i] * fnc(xi)
    except TypeError as te:
        raise NotImplementedError(
            f"{type(fnc)} function type not implemented yet. Define your function using the classic python functions definition"
        ) from te
    area *= (b - a) / 2.0
    return area


def _multicore_computation(
    n_cores: int, func: _t.AnaliticalFunc, val_dicts: list[dict[str, float]]
) -> _t.Array:
    """
    Computation of the input function using multicore parallelization

    Parameters
    ----------
    n_cores : int
        Number of cores used.
    func : sympy function
        The function to numerically compute.
    val_dicts : dict
        The dictionaries of variable-values used to compute the function.

    Returns
    -------
    computed_func : ndarray
        Array of the computed function.

    """
    compute = __compute_sympy(func)
    with _mp.Pool(n_cores) as pool:
        computed_func = pool.map(compute.compute, val_dicts)
    return computed_func


def _lambdified_computation(
    func: _t.AnaliticalFunc, variables: _t.Variables, var_data: list[_t.Array]
) -> _t.Array:
    """
    Compute the input function compiling the sympy expression using numpy, for
    way faster computation.

    Parameters
    ----------
    func : sympy function
        The function to compute.
    variables : list of sympy symbols
        The variables of the function.
    var_data : ndarray
        The values for the variables of the function.

    Returns
    -------
    computed_func : ndarray
        The result computed function.
    """
    f_lambdified = _sp.lambdify(variables, func, modules="numpy")
    if not isinstance(var_data, list):
        var_data = [var_data]
    result = f_lambdified(*var_data)
    computed_func = []
    for x in result:
        computed_func.append(float(x))
    return computed_func


def _data_dict_creation(
    variables: _t.Variables, var_data: list[_t.Array]
) -> list[dict[str, float]]:
    """
    function which creates the list of dictionaries in the format needed to compute
    sympy functions.

    Parameters
    ----------
    variables : list of sympy symbols
        The variables of the function to compute.
    var_data : ndarray
        the values for the variables of the function to compute.

    Returns
    -------
    val_dicts : list of dict
        A list of dictionaries, each containing the variables with an associated
        value, to be passed to the function in order to be computed.
    """
    N = _np.array(var_data).shape[-1]
    val_dicts = []
    for n in range(N):
        data_dict = {}
        for i, var in enumerate(variables):
            var_name = f"{var}"
            data_dict[var_name] = var_data[i][n]
        val_dicts.append(data_dict)
    return val_dicts


class __compute_sympy:
    """
    Sub-Class for multiprocessing computation
    """

    def __init__(self, func: _t.AnaliticalFunc):
        self.f = func

    def compute(self, vals: dict[str, float]) -> float:
        return float(_sp.N(self.f.subs(vals)))
