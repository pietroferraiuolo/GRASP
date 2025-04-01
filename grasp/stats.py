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
from astropy.table import Table as _Table
from astroML.density_estimation import XDGMM
from grasp.core.folder_paths import R_SOURCE_FOLDER as _RSF
from grasp.analyzers._Rcode import check_packages as _checkRpackages, r2py_models as _rm
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


def regression(data, kind="gaussian", verbose: bool = True):
    """
    Regression model estimation function.

    This function fits the input data to an analythical function. The regression model
    can be of different types, such as linear, or gaussian regression.
    The function uses the `minpack.lm` R package to fit the model.

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
    model = _rm.RegressionModel(regression_model, type=kind)
    _np2r.deactivate()
    return model


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
