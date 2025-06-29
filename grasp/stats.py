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
from grasp import types as _T
from astropy.table import Table as _Table
from astroML.density_estimation import XDGMM
from grasp.core.folder_paths import R_SOURCE_FOLDER as _RSF
from grasp.analyzers._Rcode import check_packages as _checkRpackages, r2py_models as _rm
from scipy.optimize import curve_fit as _curve_fit
import rpy2.robjects as _ro
from rpy2.robjects import (
    r as _R,
    numpy2ri as _np2r,
    pandas2ri as _pd2r,
    globalenv as _genv,
)


def XD_estimator(
    data: _T.Array,
    errors: _T.Array,
    correlations: _T.Optional[dict[tuple[int, int], _T.Array]] = None,
    **xdargs: tuple[str, ...],
):
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
    errors = _data_format_check(errors)  # (n_samples, n_features)
    if correlations is not None:
        covariance_matrix = _construct_covariance_matrices(errors, correlations)
    else:
        covariance_matrix = _np.array(
            [_np.diag(e**2) for e in errors]
        )  # (n_samples, n_features, n_features)
    model = XDGMM(random_state=_seed(), **xdargs)
    model.fit(x_data, covariance_matrix)
    return model


def kfold_gmm_estimator(
    data: _T.ArrayLike, folds: int, **gmm_params: dict[str, _T.Any]
) -> _T.GMModel:
    """
    K-Fold Gaussian Mixture Model Estimation function.

    This function fits a K-Fold Gaussian Mixture Model (GMM) to the input data.
    The GMM is a probabilistic model that can be used to estimate the underlying
    distribution of a dataset. The function uses the `mclust` R package to fit the
    model.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be fitted with a gaussian mixture model, of shape `[n_samples, n_features]`.
    folds : int
        The number of folds for cross-validation.
    **gmm_params : optional
        Additional keyword arguments for the R GM_model function. See
        <a href="https://cran.r-project.org/web/packages/mclust/mclust.pdf">
        mclust</a> documentation for more information.

    Returns
    -------
    fitted_model : grasp.KFoldGMM
        The fitted gaussian mixture model and its parameters.
    """
    _checkRpackages("mclust")
    _np2r.activate()
    code = _os.path.join(_RSF, "gaussian_mixture.R")
    _R(f'source("{code}")')
    r_data = _np2r.numpy2rpy(data)
    # Convert kwargs to R list
    r_kwargs = _ro.vectors.ListVector(gmm_params)
    # Call the R function with the data and additional arguments
    fitted_model = _genv["KFoldGMM"](r_data, folds, **dict(r_kwargs.items()))
    _np2r.deactivate()
    return _rm.KFoldGMM(fitted_model)


def gaussian_mixture_model(
    train_data: _T.Array,
    fit_data: _T.Optional[_T.Array] = None,
    **kwargs: dict[str, _T.Any],
) -> _T.GMModel:
    """
    Gaussian Mixture Estimation function.

    This function fits a Gaussian Mixture Model (GMM) to the input data.
    The GMM is a probabilistic model that can be used to estimate the underlying
    distribution of a dataset. The function uses the `mclust` R package to fit the
    model.

    Parameters
    ----------
    trin_data : numpy.ndarray
        The data to be fitted with a gaussian mixture model, of shape `[n_samples, n_features]`.
    fit_data : numpy.ndarray, optional
        The data to be fitted with a gaussian mixture model, of shape `[n_samples, n_features]`.
        If provided, the fitted model will perform predictions to this data.
    **kwargs : optional
        Additional keyword arguments for the R GM_model function. See
        <a href="https://cran.r-project.org/web/packages/mclust/mclust.pdf">
        mclust</a> documentation for more information.

    Returns
    -------
    fitted_model : grasp.GaussianMixtureModel
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
        fitted_model = _genv["GMMTrainAndFit"](
            r_data, r_fit_data, **dict(r_kwargs.items())
        )
        clusters = fitted_model.rx2("cluster")
        fitted_model = fitted_model.rx2("model")
    else:
        r_data = _np2r.numpy2rpy(train_data)
        # Convert kwargs to R list
        r_kwargs = _ro.vectors.ListVector(kwargs)
        # Call the R function with the data and additional arguments
        fitted_model = _genv["GaussianMixtureModel"](r_data, **dict(r_kwargs.items()))
        clusters = None
    _np2r.deactivate()
    return _rm.GaussianMixtureModel(fitted_model, clusters)


def fit(
    data: _T.Array,
    method: _T.FittingFunc = "gaussian",
    fit_type: str = "distribution",
    **kwargs: dict[str, _T.Any],
) -> _T.RegressionModels:
    """
    General fitting function that combines distribution fitting and raw data fitting.

    For in depth information on the fitting methods, please refer to the
    `fit_distribution` and `fit_data_points` functions.
    This function allows you to choose between fitting the distribution of the data
    or fitting the raw data points. The fitting method is determined by the `fit_type`
    parameter. The function will call the appropriate fitting function based on the
    specified `fit_type`.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be analyzed.
    method : str or function
        The type of regression model to be fitted. If passed as str, options are:<br>
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

        Only for the `datapoint`, a custom fitting function can be passed as a
        callable. Also, the fitting type:<br>
        - "polyN" (where N is the degree of the polynomial)

        is added to the list of available options.<br>
        If a custom fitting function is passed, it must be a callable function
        of the form:
        ```python
        def custom_function(x, *params):
            # Your custom fitting function here
            return y
        ```
        where `x` is the independent variable data and `params` are the parameters
        to be fitted. The function should return the dependent variable data.
        The function can also be a `lambda` function.
    fit_type : str, optional
        The type of fitting to perform. Options are:
        - "distribution": Fits the data's distribution, performing a KDE
            (calls `fit_distribution`).
        - "datapoint": Fits the data points (calls `fit_points`).
    **kwargs : optional
        Additional arguments passed to the respective fitting functions.
        Check the documentation of `fit_distribution` and `fit_data_points`
        for more information on the available arguments.

    Returns
    -------
    model : RegressionModel or PyRegressionModel
        The fitted model.
    """
    if fit_type == "distribution":
        return fit_distribution(data=data, method=method, **kwargs)
    elif fit_type == "datapoint":
        return fit_data_points(data=data, method=method, **kwargs)
    else:
        raise ValueError(
            f"Invalid fit_type: {fit_type}. Choose 'distribution' or 'datapoint'."
        )


def fit_distribution(
    data: _T.Array,
    bins: str | int | _T.Array = "detailed",
    method: str = "gaussian",
    verbose: bool = True,
    plot: bool = False,
) -> _T.RegressionModels:
    """
    Regression model estimation function.

    This function fits the input data **_distribution_** to an analythical function.
    The regression model can be of different types, such as linear, or gaussian
    regression. The function uses the `minpack.lm` R package to fit the model.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be analyzed.
    method : str, optional
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
    if method == "linear":
        DeprecationWarning(
            "The 'linear' method in `fit_distribution is deprecated. Use `grasp.stats.fit_data_points(..., method='linear')` instead."
        )
        _pd2r.activate()
        reg_func = _genv["linear_regression"]
        D = _np.array(data).shape[-1] if isinstance(data, list) else data.shape[-1]
        if D != 2:
            x = list(_np.arange(0.0, 1.0, 1 / len(data)))
            y = list(data)
            data = _pd.DataFrame({"x": x, "y": y})
        r_data = _pd2r.py2rpy_pandasdataframe(data)
        _pd2r.deactivate()
        regression_model = reg_func(r_data, verb=verbose)
    else:
        reg_func = _genv["regression"]
        if isinstance(bins, str) and bins == "knuth":
            # print("knuth?") # DEBUG
            from astropy.stats import knuth_bin_width

            _, bins = knuth_bin_width(data, return_bins=True)
            bins = _np2r.numpy2rpy(bins)
        elif isinstance(bins, int):
            # print("int?") # DEBUG
            bins = _ro.IntVector([bins])
        elif isinstance(bins, (list, _np.ndarray)):
            # print("array?") # DEBUG
            bins = _np2r.numpy2rpy(bins)
        r_data = _np2r.numpy2rpy(data)
        regression_model = reg_func(r_data, method=method, bins=bins, verb=verbose)
    model = _rm.RegressionModel(regression_model, kind=method)
    _np2r.deactivate()
    if plot:
        from grasp.plots import regressionPlot

        regressionPlot(model)
    return model


def fit_data_points(
    data: _T.Optional[_T.Array] = None,
    x_data: _T.Optional[_T.Array] = None,
    method: _T.FittingFunc = "linear",
    plot: bool = False,
    *curvefit_args: tuple[str, ...],
) -> _T.RegressionModels:
    """
    This function fits the imput **data** (not its distribution, see `regression`)
    to an analythical function.

    It offers several pre-built functions to fit the data, such as:
    - linear
    - polyN (where N is the degree of the polynomial)
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
    >     return a * np.exp(b * x)
    > grasp.stats.fit_data(x, y, exponential)
    > # or, equivalently
    > grasp.stats.fit_data(x, y, lambda x, a, b: a * np.exp(b * x))
    ```

    Parameters
    ----------
    data : numpy.ndarray
        The y data to be fitted.
    x_data : numpy.ndarray, optional
        The indipendent variable data.
    method : str or function
        The function to be used for the fitting. If a string, it must be one of the
        pre-defined functions listed above. If a function, it must be a callable
        function that takes the x_data as input and returns the y_data (can be
        a `lambda` function).
    plot : bool, optional
        If True, plot the fitted data.

    Additional Parameters
    ---------------------
    *curvefit_args : additional parameters of the `scipy.optimize.curve_fit` function (See
    <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html">scipy</a>
    documentation for more details).<br>
    **NOTE**: The `curvefit_args` won't work if the fitting method is `polyN`, as this
    function uses the numpy `polyfit` and `polyval` functions to perform the fitting
    on the data.

    Returns
    -------
    fitted: PyRegressionModel
        The fitted model as a PyRegressionModel object. The object contains the following attributes:
        - data: the original data to be fitted
        - x: the indipendent variable data
        - y_fit: the result fitted y data
        - parameters: the coefficients of the fitted function
        - residuals: the residuals of the fit1
        - covariance: the covariance matrix of the fitted parameters
        - kind: the type of fit performed (e.g. linear, gaussian, etc.)

    """
    method_is_str = isinstance(method, str)
    if data is None:
        fitted = {
            "data": None,
            "x": None,
            "y_fit": None,
            "parameters": None,
            "residuals": None,
            "covmat": None,
        }
        return _rm.PyRegressionModel(fitted, "")
    x_data = (
        _np.arange(_np.finfo(_np.float32).eps, 1.0, 1 / len(data))
        if x_data is None
        else x_data
    )
    if method_is_str and "poly" in method:
        from numpy import polyfit, polyval

        try:
            degree = int(method[-2:])
        except ValueError:
            degree = int(method[-1])
        coeffs = polyfit(x_data, data, degree)
        y_fit = polyval(coeffs, x_data)
        residuals = data - y_fit
        fitted = {
            "data": data,
            "x": x_data,
            "y_fit": y_fit,
            "parameters": coeffs,
            "residuals": residuals,
            "covmat": "unavailable",
        }
        model = _rm.PyRegressionModel(fitted, f"{degree}-degree polynomial")
    else:
        f = method if callable(method) else _get_function(method)
        if not callable(f):
            raise ValueError(
                f"The provided function argument `f` must be either a string or a callable: {type(f)}"
            )
        popt, pcov, infodict, _, _ = _curve_fit(
            f, x_data, data, full_output=True, *curvefit_args
        )
        y_fit = f(x_data, *popt)
        fitted = {
            "data": data,
            "x": x_data,
            "y_fit": y_fit,
            "parameters": popt,
            "residuals": infodict["fvec"],
            "covmat": pcov,
        }
        kind = method if method_is_str else "custom"
        model = _rm.PyRegressionModel(fitted, kind)
    if plot:
        from grasp.plots import regressionPlot

        regressionPlot(model, f_type="datapoint")
    return model


def bootstrap_statistic(
    data: _T.Array,
    statistic_function: _T.Callable[..., _T.Any],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.68,
    method: str = "percentile",
    parallel: bool = False,
    n_jobs: _T.Optional[int] = None,
    show_progress: bool = True,
    return_bootstrap_distribution: bool = False,
    random_seed: _T.Optional[int] = None,
    *args: _T.Any,
    **kwargs: _T.Any,
) -> tuple[float, float, _T.Optional[_T.Array]]:
    """
    Calculate any statistic with uncertainty using bootstrap resampling,
    optimized for astrophysical data analysis.

    Parameters
    ----------
    data : np.ndarray
        Input data array. Can be 1D or 2D (for multivariate data).
    statistic_function : callable
        Function that computes the desired statistic. Should accept the
        data array as first argument and return a scalar or array.
    n_bootstrap : int, optional
        Number of bootstrap iterations. Default is 1000.
    confidence_level : float, optional
        Confidence level for uncertainty estimation (between 0 and 1).
        Default is 0.68 (roughly equivalent to 1-sigma).
    method : str, optional
        Method for computing confidence intervals:
        - 'percentile': Uses percentiles of bootstrap distribution
        - 'std': Uses standard deviation of bootstrap distribution
        Default is 'percentile'.
    parallel : bool, optional
        Whether to use parallel processing. Default is False.
    n_jobs : int, optional
        Number of processes to use for parallel execution.
        If None, uses all available CPUs. Default is None.
    show_progress : bool, optional
        Whether to display a progress bar. Default is True.
    return_bootstrap_distribution : bool, optional
        Whether to return the bootstrap distribution. Default is False.
    random_seed : int, optional
        Random seed for reproducibility. Default is None.
    *args, **kwargs
        Additional arguments passed to statistic_function.

    Returns
    -------
    (statistic_value, uncertainty, bootstrap_distribution) : if return_bootstrap_distribution=True
    (statistic_value, uncertainty) : otherwise

    Examples
    --------
    ```python
    import numpy as np
    # Calculate median and its uncertainty
    data = np.random.normal(100, 15, 50)  # Simulated astronomical measurements
    median, uncertainty = bootstrap_statistic(data, np.median)
    print(f"Median: {median:.1f} ± {uncertainty:.1f}")
    # Calculate mean and its asymmetric uncertainty (for skewed distributions)
    from astropy.stats import sigma_clip
    mean, uncertainty, dist = bootstrap_statistic(
        data, sigma_clip, method='percentile', confidence_level=0.95,
        return_bootstrap_distribution=True)
    lower, upper = np.percentile(dist, [2.5, 97.5])  # 95% confidence interval
    print(f"Mean: {mean:.1f} +{upper-mean:.1f} -{mean-lower:.1f}")
    ```
    """
    from tqdm import tqdm as _tqdm
    from multiprocessing import Pool as _Pool, cpu_count as _ncpu

    # Input validation
    data = _np.asarray(data)
    if data.size == 0:
        raise ValueError("Input data array is empty")

    if n_bootstrap <= 0:
        raise ValueError("Number of bootstrap iterations must be positive")

    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1")

    if method not in ["percentile", "std"]:
        raise ValueError("Method must be either 'percentile' or 'std'")

    # Set random seed for reproducibility if specified
    if random_seed is not None:
        _np.random.seed(random_seed)

    # Try to compute the statistic on the original data to check if it works
    try:
        original_statistic = statistic_function(data, *args, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to compute statistic on input data: {str(e)}")

    # Determine array shape and setup bootstrap samples
    n_samples = data.shape[0]
    bootstrap_statistics = _np.zeros((n_bootstrap,) + _np.shape(original_statistic))

    # Generate bootstrap indices in advance
    all_indices = [
        _np.random.choice(n_samples, size=n_samples, replace=True)
        for _ in range(n_bootstrap)
    ]

    # Run bootstrap iterations
    if parallel and n_bootstrap > 50:  # Only use parallel for larger bootstrap sizes
        n_proc = n_jobs if n_jobs is not None else _ncpu()

        # Prepare arguments for parallel execution
        all_args = [
            (data, indices, statistic_function, args, kwargs) for indices in all_indices
        ]

        with _Pool(processes=n_proc) as pool:
            iter_func = pool.imap(_bootstrap_helper, all_args)
            if show_progress:
                iter_func = _tqdm(iter_func, total=n_bootstrap, desc="Bootstrap")

            bootstrap_statistics = _np.array(list(iter_func))
    else:
        iter_range = (
            _tqdm(range(n_bootstrap), desc="Bootstrap")
            if show_progress
            else range(n_bootstrap)
        )
        for i in iter_range:
            # Directly call the helper function with the prepared arguments
            bootstrap_statistics[i] = _bootstrap_helper(
                (data, all_indices[i], statistic_function, args, kwargs)
            )

    # Check for NaNs from failed computations
    nan_mask = _np.isnan(bootstrap_statistics)
    if _np.isscalar(original_statistic):
        nan_count = _np.sum(nan_mask)
    else:
        nan_count = _np.sum(
            nan_mask.any(axis=tuple(range(1, bootstrap_statistics.ndim)))
        )

    if nan_count > 0:
        print(f"Warning: {nan_count} bootstrap iterations failed and were excluded")
        if _np.isscalar(original_statistic):
            bootstrap_statistics = bootstrap_statistics[~nan_mask]
        else:
            bootstrap_statistics = bootstrap_statistics[
                ~nan_mask.any(axis=tuple(range(1, bootstrap_statistics.ndim)))
            ]

    if len(bootstrap_statistics) == 0:
        raise RuntimeError("All bootstrap iterations failed")

    # Calculate uncertainty
    if method == "percentile":
        alpha = (1 - confidence_level) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100

        if _np.isscalar(original_statistic):
            lower, upper = _np.percentile(
                bootstrap_statistics, [lower_percentile, upper_percentile]
            )
            uncertainty = (upper - lower) / 2  # Average of the two-sided interval
        else:
            # Handle multidimensional statistics
            uncertainty = _np.zeros_like(original_statistic)
            for idx in _np.ndindex(original_statistic.shape):
                idx_tuple = idx if len(idx) > 0 else 0
                stat_at_idx = bootstrap_statistics[(slice(None),) + idx]
                lower, upper = _np.percentile(
                    stat_at_idx, [lower_percentile, upper_percentile]
                )
                uncertainty[idx] = (upper - lower) / 2
    else:  # 'std' method
        uncertainty = _np.std(bootstrap_statistics, axis=0, ddof=1)

    if return_bootstrap_distribution:
        return original_statistic, uncertainty, bootstrap_statistics
    return original_statistic, uncertainty


# Define this helper function at module level so it can be pickled
def _bootstrap_helper(args):
    """Helper function for bootstrap resampling that can be pickled."""
    data, indices, statistic_function, func_args, func_kwargs = args
    try:
        # Sample with replacement using the provided indices
        if data.ndim == 1:
            bootstrap_sample = data[indices]
        else:
            bootstrap_sample = data[indices, :]

        # Compute the statistic
        return statistic_function(bootstrap_sample, *func_args, **func_kwargs)
    except Exception:
        # Return NaN if computation fails
        shape = _np.shape(statistic_function(data, *func_args, **func_kwargs))
        return _np.nan * _np.ones(shape) if shape else _np.nan


def _get_function(name: str):  # -> _T.FittingFunc[..., float]:
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
        "linear": lambda x, m, q: m * x + q,
        "power": lambda x, a, b: a * x**b,
        "gaussian": lambda x, A, mu, sigma: A * exp(-((x - mu) ** 2) / (2 * sigma**2)),
        "boltzmann": lambda x, A1, A2, x0, dx: (A1 - A2) / (1 + exp((x - x0) / dx))
        + A2,
        "lognormal": lambda x, A, mu, sigma: A
        / (x * sigma * sqrt(2 * pi))
        * exp(-((log(x) - mu) ** 2) / (2 * sigma**2)),
        "exponential": lambda x, a, l: a * exp(l * x),
        "poisson": lambda x, A, l: A * exp(-l) * l**x / fact(x),
        "maxwell": lambda x, a, sigma: (a / (sigma**3))
        * 4
        * _np.pi
        * x**2
        * _np.exp(-(x**2) / (2 * sigma**2)),
        "lorentzian": lambda x, a, x0, gamma: a
        * gamma
        / (2 * pi)
        / ((x - x0) ** 2 + (gamma / 2) ** 2),
        "rayleigh": lambda x, a, sigma: (a / (sigma**2))
        * x
        * _np.exp(-(x**2) / (2 * sigma**2)),
    }
    return functions.get(name)


def _data_format_check(data: _T.Array | _T.TabularData) -> _T.Array:
    """
    Function which checks and formats the input data to be ready
    for the XDGMM model fit.

    Parameters:
    ----------
    data : ArrayLike, astropy.table.Table, pandas.DataFrame
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


def _construct_covariance_matrices(
    errors: _T.Array, correlations: dict[tuple[int, int], _T.Array]
) -> _T.Array:
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
