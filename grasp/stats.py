"""
Author(s)
---------
- Pietro Ferraiuolo : Written in 2024

Description
-----------
This module provides a series of functions for the statistical analysis of
(astronomical) data. The functions are designed to be used in the context of
the grasp software. The module comes with a related sub-module, `r2py_models.py`,
which handles the conversion of R objects to Python objects, since the majority
of the statistical analysis is done through R scripts.

"""

import os as _os
import numpy as _np
import time as _time
import pandas as _pd
import rpy2.robjects as _ro
from typing import Callable
from astropy.table import Table as _Table
from astroML.density_estimation import XDGMM
from grasp.core.folder_paths import R_SOURCE_FOLDER as _RSF
from grasp.analyzers._Rcode import check_packages as _checkRpackages, r2py_models as _rm
from scipy.optimize import curve_fit
from rpy2.robjects import (
    r as _R,
    numpy2ri as _np2r,
    pandas2ri as _pd2r,
    globalenv as _genv
)


def XD_estimator(data, errors, correlations=None, *xdargs):
    """
    Extreme Deconvolution Estimation function.

    This function fits an eXtreme Deconvolution Gaussian Mixture Model (XDGMM)
    to the input data. The XDGMM, part of the astroML package, is a probabilistic
    model that can be used to estimate the underlying distribution of a dataset,
    taking into account measurement errors and correlations between features.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be analyzed.
    errors : numpy.ndarray
        The errors associated with the data.
    correlations : dict
        The correlations between the data. If a dictionary, the keys are
        (i, j) tuples representing feature pairs and the values are
        (n_samples,) arrays of correlation coefficients for each sample.
    *xdargs : optional
        XDGMM hyper-parameters for model tuning. See
        <a href="https://www.astroml.org/modules/generated/astroML.density_estimation.XDGMM.html">
        astroML</a> documentation for more information.

    Returns
    -------
    model : astroML.density_estimation.XDGMM
        The XDGMM fitted model.
    """
    x_data = _data_format_check(data)  # (n_samples, n_features)
    errors = _data_format_check(errors)  # (n_samples, n_features, n_features)
    if correlations is not None:
        covariance_matrix = _construct_covariance_matrices(errors, correlations)
    else:
        covariance_matrix = _np.array([_np.diag(e**2) for e in errors])
    model = XDGMM(random_state=_seed(), *xdargs)
    model.fit(x_data, covariance_matrix)
    return model


def gaussian_mixture_model(train_data, fit_data=None, **kwargs):
    """
    Gaussian Mixture Estimation function.

    This function fits a Gaussian Mixture Model (GMM) to the input data.
    The GMM is a probabilistic model that can be used to estimate the underlying
    distribution of a dataset. The function uses the `mclust` R package to fit the
    model.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be fitted with a gaussian mixture model.
    **kwargs : optional
        Additional keyword arguments for the R GM_model function. See
        <a href="https://cran.r-project.org/web/packages/mclust/mclust.pdf">
        mclust</a> documentation for more information.

    Returns
    -------
    fitted_model : dict
        The fitted gaussian mixture model and its parameters.
    """
    _checkRpackages("mclust")
    _np2r.activate()
    code = _os.path.join(_RSF, "gaussian_mixture.R")
    _R(f'source("{code}")')
    if fit_data is not None:
        r_data = _np2r.numpy2rpy(train_data)
        r_fit_data = _np2r.numpy2rpy(fit_data)
        # Convert kwargs to R list
        r_kwargs = _ro.vectors.ListVector(kwargs)
        # Call the R function with the data and additional arguments
        fitted_model = _genv["GaussianMixtureModel"](
            r_data, r_fit_data, **dict(r_kwargs.items())
        )
        clusters = fitted_model.rx2("cluster")
        fitted_model = fitted_model.rx2("model")
    else:
        r_data = _np2r.numpy2rpy(train_data)
        # Convert kwargs to R list
        r_kwargs = _ro.vectors.ListVector(kwargs)
        # Call the R function with the data and additional arguments
        fitted_model = _genv["GM_model"](r_data, **dict(r_kwargs.items()))
        clusters = None
    _np2r.deactivate()
    return _rm.GMModel(fitted_model, clusters)


def regression(data, kind="gaussian", verbose: bool = True, plot: bool = False):
    """
    Regression model estimation function.

    This function fits the input data **_distribution_** to an analythical function.
    The regression model can be of different types, such as linear, or gaussian
    regression. The function uses the `minpack.lm` R package to fit the model.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be analyzed.
    kind : str, optional
        The type of regression model to be fitted. Options are:
        - "linear",
        - "power",
        - "gaussian",
        - "poisson",
        - "lognormal",
        - "exponential",
        - "boltzmann",
        - "king",
        - "maxwell",
        - "lorentzian",
        - "rayleigh",
    verbose : bool, optional
        If True, print verbose output from the fitting routine.

    Returns
    -------
    model : grasp.analyzers._Rcode.RegressionModel
        The fitted regression model, translated from R to Py.
    """
    _checkRpackages("minpack.lm")
    _np2r.activate()
    regression_code = _os.path.join(_RSF, "regression.R")
    _R(f'source("{regression_code}")')
    if kind=='linear':
        _pd2r.activate()
        reg_func = _genv["linear_regression"]
        D = _np.array(data).shape[-1] if isinstance(data, list) else data.shape[-1]
        if D != 2:
            x = list(_np.arange(0.,1.,1/len(data)))
            y = list(data)
            data = _pd.DataFrame({'x': x, 'y': y})
        r_data = _pd2r.py2rpy_pandasdataframe(data)
        _pd2r.deactivate()
    else:
        reg_func = _genv["regression"]
        r_data = _np2r.numpy2rpy(data)
    regression_model = reg_func(r_data, method=kind, verb=verbose)
    model = _rm.RegressionModel(regression_model, kind=kind)
    _np2r.deactivate()
    if plot:
        from grasp.plots import regressionPlot
        regressionPlot(model)
    return model


def fit_data(y_data, fit: str | Callable, x_data = None, y_err = None, plot: bool = False):
    """
    This function fits the imput **data** (not its distribution, see `regression`)
    to an analythical function.

    It offers several pre-built functions to fit the data, such as:
    - linear
    - power
    - gaussian
    - poisson
    - lognormal
    - exponential
    - boltzmann
    - maxwell
    - lorentzian
    - rayleigh

    but a custom function can be provided as well.

    **NOTE**: for a custom fitting function to be provided, it must be python-defined

    ```python
    > grasp.stats.fit_data(x, y, 'exponential')
    > # The above call is equivalent to doing:
    > def exponential(x, a, b):
    >     return a * _np.exp(b * x)
    > grasp.stats.fit_data(x, y, exponential)
    ```

    Parameters
    ----------
    y_data : numpy.ndarray
        The y data to be fitted.
    f : str or function
        The function to be used for the fitting. If a string, it must be one of the
        pre-defined functions listed above. If a function, it must be a callable
        function that takes the x_data as input and returns the y_data (can be
        a `lambda` function).
    x_data : numpy.ndarray, optional
        The indipendent variable data.
    y_err : numpy.ndarray, optional
        The errors associated with the y_data. If provided, the fitting will
        take into account the errors and will return the covariance matrix of
        the fitted parameters.

    Returns
    -------
    fitted: dict
        A dictionary containing the fitted parameters and the covariance matrix:
    - data : numpy.ndarray
        The original y data.
    - x : numpy.ndarray
        The x data used for the fitting. If it has been passed, it's simply
        `x_data`
    - parameters : numpy.ndarray
         The optimal values of the parameters for the fitted function.
    - covariance : numpy.ndarray
         The covariance matrix of the fitted parameters.

    """
    x_data = _np.arange(_np.finfo(_np.float32).eps,1.,1/len(y_data)) if x_data is None else x_data
    f = fit if isinstance(fit, Callable) else _get_function(fit)
    if not callable(f):
        raise ValueError(
            f"The provided function argument `f` must be either a string or a callable: {type(f)}"
        )
    if y_err is not None:
        popt, pcov = curve_fit(f, x_data, y_data, sigma=y_err, absolute_sigma=True)
    else:
        popt, pcov = curve_fit(f, x_data, y_data)
    y_fit = f(x_data, *popt)
    residuals = y_data - y_fit
    fitted = {
        'data': y_data,
        'x': x_data,
        'y_fit': y_fit,
        "parameters": popt,
        "residuals": residuals,
        "covariance": pcov
    }
    kind = fit if isinstance(fit, str) else "custom"
    model = _rm.PyRegressionModel(fitted, kind)
    if plot:
        from grasp.plots import regressionPlot
        regressionPlot(model)
    return model


def _get_function(name: str):
    """
    This function returns the function corresponding to the provided name.
    The function must be defined in the `grasp.stats` module.

    Parameters
    ----------
    name : str
        The name of the function to be retrieved.

    Returns
    -------
    function
        The function corresponding to the provided name.
    """

    exp = _np.exp
    pi = _np.pi
    log = _np.log
    sqrt = _np.sqrt
    fact = _np.math.factorial
    functions = {
        "linear"      : lambda x, m, q           : m * x + q,
        "power"       : lambda x, a, b           : a * x**b,
        "gaussian"    : lambda x, A, mu, sigma   : A * exp(-(x - mu)**2 / (2 * sigma**2)),
        "boltzmann"   : lambda x, A1, A2, x0, dx : (A1 - A2) / (1 + exp((x - x0) / dx)) + A2,
        "lognormal"   : lambda x, A, mu, sigma   : A / (x * sigma * sqrt(2 * pi)) * exp(-(log(x) - mu)**2 / (2 * sigma**2)),
        "exponential" : lambda x, a, l           : a * exp(l * x),
        "poisson"     : lambda x, A, l           : A * exp(-l) * l**x / fact(x),
        "maxwell"     : lambda x, a, sigma       : (a / (sigma**3)) * 4 * _np.pi * x**2 * _np.exp(-x**2 / (2 * sigma**2)),
        "lorentzian"  : lambda x, a, x0, gamma   : a * gamma / (2 * pi) / ((x - x0)**2 + (gamma / 2)**2),
        "rayleigh"    : lambda x, a, sigma       : (a / (sigma**2)) * x * _np.exp(-x**2 / (2 * sigma**2)),
    }
    return functions.get(name)


def _data_format_check(data):
    """
    Function which checks and formats the input data to be ready
    for the XDGMM model fit.

    Parameters:
    ----------
    data : numpy.ndarray, list, astropy.table.Table, pandas.DataFrame
        The data whose format has to be checked.

    Returns:
    -------
    data : numpy.ndarray
        The data in the correct format for the XDGMM model.
        Rturned in shape (n_samples, n_features).
    """
    if isinstance(data, (_Table, _pd.DataFrame)):
        data = data.to_numpy()
    elif isinstance(data, list):
        data = _np.stack(data).T
    return data


def _construct_covariance_matrices(errors, correlations: dict):
    """
    Constructs covariance matrices for each sample based on given errors and correlations.

    Parameters:
    ----------
    errors : numpy.ndarray
        An (n_samples, n_features) array with standard deviations for each feature.
    correlations : dict
        A dictionary of correlation arrays where keys are (i, j) tuples
        representing feature pairs and values are (n_samples,) arrays of
        correlation coefficients for each sample.

    Returns:
    -------
    cov_tensor : numpy.ndarray
        Array of covariance matrices, one for each sample.
    """
    if not isinstance(correlations, dict):
        raise ValueError(
            f"Correlations must be a dictionary, not {type(correlations)}."
        )
    n_samples, n_features = errors.shape
    X_error = []
    for i in range(n_samples):
        # Initialize covariance matrix for sample i
        cov_matrix = _np.zeros((n_features, n_features))
        # Set variances on the diagonal
        for j in range(n_features):
            cov_matrix[j, j] = errors[i, j] ** 2
        # Set covariances for each pair based on correlations
        for (f1, f2), rho in correlations.items():
            cov_matrix[f1, f2] = rho[i] * errors[i, f1] * errors[i, f2]
            cov_matrix[f2, f1] = cov_matrix[f1, f2]  # symmetry
        X_error.append(cov_matrix)
    cov_tensor = _np.array(X_error)
    return cov_tensor


def _seed():
    return int(_time.time())
