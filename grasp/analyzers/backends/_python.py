"""Pure-Python implementations of the historical R routines.

Authors
-------
- Pietro Ferraiuolo : written in 2024 (R originals)
- GRASP Phase 3 cleanup : Python ports (2026)

Notes
-----
This module replaces ``grasp/analyzers/_Rcode/regression.R`` and
``grasp/analyzers/_Rcode/gaussian_mixture.R``. It uses

- :class:`lmfit.Model` for non-linear least-squares fits (gaussian,
  boltzmann, exponential, king, maxwell, rayleigh, lorentzian, power,
  lognormal). :func:`scipy.optimize.curve_fit` is used as a fallback
  when ``lmfit`` is unavailable.
- :class:`statsmodels.api.OLS` for the linear regression, exposing
  p-values and the full model summary.
- :class:`sklearn.mixture.GaussianMixture` and
  :class:`sklearn.model_selection.KFold` for the (single and K-fold)
  Gaussian mixture model. CV is scored via :meth:`GaussianMixture.score`
  (mean log-likelihood) and BIC via :meth:`GaussianMixture.bic`.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from grasp.utils.rng import default_rng

_logger = logging.getLogger("grasp.analyzers.backends._python")


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def _gaussian(x, A, mean, sd):
    return A * np.exp(-((x - mean) ** 2) / (2 * sd**2))


def _boltzmann(x, A1, A2, x0, dx):
    return (A1 - A2) / (1 + np.exp((x - x0) / dx)) + A2


def _exponential(x, a, b):
    # Decay form, matching ``regression.R`` and ``grasp.stats._get_function``.
    return a * np.exp(-b * x)


def _power(x, a, b):
    return a * np.power(x, b)


def _lognormal(x, A, mu, sigma):
    return A / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(
        -((np.log(x) - mu) ** 2) / (2 * sigma**2)
    )


def _maxwell(x, A, sigma):
    return A * x**2 * np.exp(-(x**2) / (2 * sigma**2))


def _rayleigh(x, A, sigma):
    return A * x * np.exp(-(x**2) / (2 * sigma**2))


def _lorentzian(x, A, x0, gamma):
    return A * gamma / (2 * np.pi) / ((x - x0) ** 2 + (gamma / 2) ** 2)


def _king(x, A, sigma, ve):
    out = A * (np.exp(-(x**2) / (2 * sigma**2)) - np.exp(-(ve**2) / (2 * sigma**2)))
    return np.where(x <= ve, out, 0.0)


_REGRESSION_MODELS: dict[str, dict[str, Any]] = {
    "gaussian": {
        "fn": _gaussian,
        "param_names": ("a", "mean", "sd"),
        "param_order": ("A", "mean", "sd"),
        "start": lambda x, y: {"A": float(np.max(y)), "mean": float(np.mean(x)), "sd": float(np.std(x) or 1.0)},
    },
    "boltzmann": {
        "fn": _boltzmann,
        "param_names": ("A1", "A2", "x0", "dx"),
        "param_order": ("A1", "A2", "x0", "dx"),
        "start": lambda x, y: {
            "A1": float(np.max(y)),
            "A2": float(np.min(y)),
            "x0": 0.0,
            "dx": 1.0,
        },
    },
    "exponential": {
        "fn": _exponential,
        "param_names": ("a", "b"),
        "param_order": ("a", "b"),
        "start": lambda x, y: {"a": float(np.max(y)), "b": 0.0},
    },
    "power": {
        "fn": _power,
        "param_names": ("a", "b"),
        "param_order": ("a", "b"),
        "start": lambda x, y: {"a": float(np.max(y)), "b": 1.0},
    },
    "lognormal": {
        "fn": _lognormal,
        "param_names": ("A", "mu", "sigma"),
        "param_order": ("A", "mu", "sigma"),
        "start": lambda x, y: {
            "A": float(np.max(y)),
            "mu": float(np.mean(np.log(np.clip(x, 1e-12, None)))),
            "sigma": float(np.std(x) or 1.0),
        },
    },
    "maxwell": {
        "fn": _maxwell,
        "param_names": ("A", "sigma"),
        "param_order": ("A", "sigma"),
        "start": lambda x, y: {"A": float(np.max(y)), "sigma": float(np.std(x) or 1.0)},
    },
    "rayleigh": {
        "fn": _rayleigh,
        "param_names": ("A", "sigma"),
        "param_order": ("A", "sigma"),
        "start": lambda x, y: {"A": float(np.max(y)), "sigma": float(np.std(x) or 1.0)},
    },
    "lorentzian": {
        "fn": _lorentzian,
        "param_names": ("A", "x0", "gamma"),
        "param_order": ("A", "x0", "gamma"),
        "start": lambda x, y: {
            "A": float(np.max(y)),
            "x0": float(np.mean(x)),
            "gamma": float(np.std(x) or 1.0),
        },
    },
    "king": {
        "fn": _king,
        "param_names": ("A", "sigma", "ve"),
        "param_order": ("A", "sigma", "ve"),
        "start": lambda x, y: {
            "A": float(np.max(y)),
            "sigma": float(np.std(x) or 1.0),
            "ve": float(np.min(x)),
        },
    },
}


def _bin_edges(data: np.ndarray, bins: Union[str, int, np.ndarray]) -> np.ndarray:
    """Compute histogram bin edges according to the regression.R contract."""

    if isinstance(bins, str):
        if bins == "detailed":
            n = int(np.ceil(1.5 * np.sqrt(len(data))))
            return np.linspace(np.min(data), np.max(data), n)
        if bins == "knuth":
            from astropy.stats import knuth_bin_width

            _, edges = knuth_bin_width(data, return_bins=True)
            return edges
        raise ValueError(f"Unknown bins spec: {bins!r}")
    if isinstance(bins, int):
        return np.linspace(np.min(data), np.max(data), bins + 1)
    edges = np.asarray(bins, dtype=float)
    edges[0] = np.min(data)
    edges[-1] = np.max(data)
    return edges


def fit_distribution_python(
    data: np.ndarray,
    *,
    method: str,
    bins: Union[str, int, np.ndarray] = "detailed",
    use_lmfit: bool = True,
) -> PyFitResult:
    """Histogram-based non-linear regression (Python port of regression.R).

    Parameters
    ----------
    data : ndarray
        1-D array of observations to fit a distribution to.
    method : str
        One of ``"gaussian"``, ``"boltzmann"``, ``"exponential"``,
        ``"power"``, ``"lognormal"``, ``"maxwell"``, ``"rayleigh"``,
        ``"lorentzian"`` or ``"king"``.
    bins : str, int or ndarray, optional
        Histogram bin specification. Strings ``"detailed"`` and
        ``"knuth"`` are accepted, as are explicit integers or bin edges.
    use_lmfit : bool, optional
        Use :mod:`lmfit` (default) when available. Otherwise falls back
        to :func:`scipy.optimize.curve_fit`.
    """

    if method not in _REGRESSION_MODELS:
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Choices: {sorted(_REGRESSION_MODELS)}"
        )

    data = np.asarray(data, dtype=float).ravel()
    edges = _bin_edges(data, bins)
    counts, edges_out = np.histogram(data, bins=edges)
    x = (edges_out[:-1] + edges_out[1:]) / 2.0
    y = counts.astype(float)

    spec = _REGRESSION_MODELS[method]
    fn = spec["fn"]
    p0 = spec["start"](x, y)

    if use_lmfit:
        try:
            import lmfit
        except ImportError:
            use_lmfit = False
            _logger.info("lmfit not available, falling back to scipy.optimize.curve_fit.")

    if use_lmfit:
        import lmfit

        model = lmfit.Model(fn)
        params = model.make_params(**p0)
        result = model.fit(y, params=params, x=x, max_nfev=2000)
        coeffs = np.array([result.params[name].value for name in spec["param_order"]])
        y_fit = result.best_fit
        residuals = result.residual
        try:
            covmat = result.covar
        except AttributeError:
            covmat = None
    else:
        from scipy.optimize import curve_fit

        popt, pcov = curve_fit(
            fn,
            x,
            y,
            p0=list(p0.values()),
            maxfev=5000,
        )
        coeffs = np.asarray(popt)
        y_fit = fn(x, *popt)
        residuals = y - y_fit
        covmat = pcov

    return PyFitResult(
        data=data,
        x=x,
        y=y_fit,
        residuals=residuals,
        coeffs=coeffs,
        covariance_matrix=covmat,
        kind=method,
    )


# ---------------------------------------------------------------------------
# Linear regression (P3-3)
# ---------------------------------------------------------------------------


def linear_regression_python(
    x: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    """OLS linear regression with the dict interface of ``linear_regression``.

    Returns
    -------
    dict
        ``{"data": ..., "x": ..., "y": ..., "coeffs": ..., "residuals": ...,
        "kind": "linear", "p_values": ..., "rsquared": ..., "summary": ...}``.
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    try:
        import statsmodels.api as sm

        X = sm.add_constant(x)
        results = sm.OLS(y, X).fit()
        coeffs = np.asarray(results.params)
        residuals = np.asarray(results.resid)
        return {
            "data": {"x": x, "y": y},
            "x": x,
            "y": np.asarray(results.fittedvalues),
            "coeffs": coeffs,
            "residuals": residuals,
            "kind": "linear",
            "p_values": np.asarray(results.pvalues),
            "rsquared": float(results.rsquared),
            "summary": str(results.summary()),
        }
    except ImportError:  # pragma: no cover
        _logger.info(
            "statsmodels not available, falling back to numpy.linalg.lstsq."
        )
        X = np.vstack([np.ones_like(x), x]).T
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_fit = X @ coeffs
        return {
            "data": {"x": x, "y": y},
            "x": x,
            "y": y_fit,
            "coeffs": coeffs,
            "residuals": y - y_fit,
            "kind": "linear",
            "p_values": None,
            "rsquared": None,
            "summary": "",
        }


# ---------------------------------------------------------------------------
# Gaussian Mixture (P3-2)
# ---------------------------------------------------------------------------

_MCLUST_TO_SKLEARN = {
    # mclust "modelNames" -> sklearn covariance_type. Only the four basic
    # families are exposed; the others map onto the closest sklearn analogue
    # with a runtime warning.
    "EII": "spherical",  # equal volume, spherical
    "VII": "spherical",  # variable volume, spherical
    "EEI": "diag",  # diagonal, equal shape
    "VEI": "diag",
    "EVI": "diag",
    "VVI": "diag",
    "EEE": "tied",
    "EVE": "tied",
    "VEE": "tied",
    "VVE": "tied",
    "VEV": "full",
    "EEV": "full",
    "VVV": "full",
}


def _mclust_to_sklearn(model_name: Optional[str]) -> str:
    """Translate an Mclust ``modelNames`` string to sklearn ``covariance_type``.

    Defaults to ``"full"`` and emits a :class:`UserWarning` for unsupported
    inputs.
    """

    if model_name is None:
        return "full"
    out = _MCLUST_TO_SKLEARN.get(model_name)
    if out is None:
        warnings.warn(
            f"Unrecognised mclust modelNames={model_name!r}; "
            "falling back to sklearn covariance_type='full'.",
            UserWarning,
            stacklevel=2,
        )
        return "full"
    return out


@dataclass
class PyFitResult:
    """Lightweight container for Python-backend regression fits."""

    data: np.ndarray
    x: np.ndarray
    y: np.ndarray
    residuals: np.ndarray
    coeffs: np.ndarray
    covariance_matrix: Any
    kind: str

    def save(self, path: str) -> None:
        """Serialise the result to ``path`` using :mod:`joblib`."""

        import joblib

        joblib.dump(self, path)
        _logger.info("Saved fit result to %s", path)

    @classmethod
    def load(cls, path: str) -> PyFitResult:
        import joblib

        return joblib.load(path)


@dataclass
class GaussianMixturePython:
    """Wrapper around :class:`sklearn.mixture.GaussianMixture`.

    Exposes the slice of the Mclust API that GRASP actually uses
    (``loglik``, ``bic``, ``predict``, ``parameters``) so that callers can
    swap the R model for the Python one without changing their code.
    """

    model: Any  # sklearn.mixture.GaussianMixture
    train_data: np.ndarray
    cluster_assignments: Optional[np.ndarray] = None

    @property
    def loglik(self) -> float:
        return float(self.model.score(self.train_data) * len(self.train_data))

    @property
    def bic(self) -> float:
        return float(self.model.bic(self.train_data))

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "pro": np.asarray(self.model.weights_),
            "mean": np.asarray(self.model.means_),
            "covariance": np.asarray(self.model.covariances_),
        }

    @property
    def coeffs(self) -> np.ndarray:
        return np.asarray(self.model.means_)

    @property
    def means(self) -> np.ndarray:
        return np.asarray(self.model.means_)

    @property
    def covariances(self) -> np.ndarray:
        return np.asarray(self.model.covariances_)

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self.model.weights_)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(data))

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)
        _logger.info("Saved GMM to %s", path)

    @classmethod
    def load(cls, path: str) -> GaussianMixturePython:
        import joblib

        return joblib.load(path)


def gaussian_mixture_model_python(
    train_data: np.ndarray,
    fit_data: Optional[np.ndarray] = None,
    *,
    n_components: int = 2,
    model_name: Optional[str] = "VII",
    n_init: int = 10,
    max_iter: int = 1000,
    tol: float = 1e-6,
    seed: Optional[int] = None,
) -> GaussianMixturePython:
    """Fit a Gaussian Mixture Model with :mod:`sklearn`.

    Parameters mirror the R wrapper. ``model_name`` is the original
    Mclust string (``"VII"``, ``"EII"``, ...); the translation to sklearn
    ``covariance_type`` is logged.
    """

    from sklearn.mixture import GaussianMixture

    train_data = np.asarray(train_data)
    rng = default_rng(seed)
    covariance_type = _mclust_to_sklearn(model_name)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=int(rng.integers(0, 2**31 - 1)) if seed is None else int(seed),
    )
    gmm.fit(train_data)
    cluster_assignments = None
    if fit_data is not None:
        cluster_assignments = gmm.predict(np.asarray(fit_data))
    return GaussianMixturePython(
        model=gmm,
        train_data=train_data,
        cluster_assignments=cluster_assignments,
    )


@dataclass
class KFoldGMMResult:
    """Result of K-fold cross-validation for a GMM."""

    mean_loglik: float
    mean_bic: float
    logliks: list[float]
    bics: list[float]
    models: list[GaussianMixturePython]

    @property
    def n_folds(self) -> int:
        return len(self.models)

    def best_model(self) -> GaussianMixturePython:
        """Return the fold with the lowest BIC."""

        idx = int(np.argmin(self.bics))
        return self.models[idx]


def kfold_gmm_python(
    data: np.ndarray,
    folds: int,
    *,
    n_components: int = 2,
    model_name: Optional[str] = "VII",
    seed: Optional[int] = None,
    **gmm_kwargs: Any,
) -> KFoldGMMResult:
    """K-fold cross-validation of a Gaussian Mixture Model.

    Uses :class:`sklearn.model_selection.KFold` and scores each fold with
    :meth:`GaussianMixture.score` -- the **correct** mean log-likelihood
    on the held-out fold, in contrast to the R routine in
    ``gaussian_mixture.R`` which mixes posteriors with prior weights.
    """

    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import KFold

    data = np.asarray(data)
    rng = default_rng(seed)
    cv = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(rng.integers(0, 2**31 - 1)) if seed is None else int(seed),
    )
    covariance_type = _mclust_to_sklearn(model_name)

    logliks: list[float] = []
    bics: list[float] = []
    models: list[GaussianMixturePython] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(data)):
        train, test = data[train_idx], data[test_idx]
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=int(rng.integers(0, 2**31 - 1)),
            **gmm_kwargs,
        )
        gmm.fit(train)
        score = float(gmm.score(test) * len(test))  # marginal log-likelihood
        bic = float(gmm.bic(test))
        logliks.append(score)
        bics.append(bic)
        models.append(GaussianMixturePython(model=gmm, train_data=train))
        _logger.debug(
            "KFold fold=%d loglik=%g bic=%g", fold_idx, score, bic
        )

    return KFoldGMMResult(
        mean_loglik=float(np.mean(logliks)),
        mean_bic=float(np.mean(bics)),
        logliks=logliks,
        bics=bics,
        models=models,
    )


# Public API
__all__ = [
    "PyFitResult",
    "GaussianMixturePython",
    "KFoldGMMResult",
    "fit_distribution_python",
    "linear_regression_python",
    "gaussian_mixture_model_python",
    "kfold_gmm_python",
]
