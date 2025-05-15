"""
Author(s)
---------
- Pietro Ferraiuolo : Written in 2024

Description
-----------
This module containg a series of classes which are used to convert R models
into Python dictionaries. This allows for a easier and more intuitive access
to the model parameters and results.

"""

import numpy as _np
import rpy2.robjects as _ro
from rpy2.robjects import (
    pandas2ri as _pd2r,
    numpy2ri as _np2r,
    #rinterface as _ri,
    globalenv as _genv,
    r as _R,
)


class GaussianMixtureModel:
    """
    Class to convert the R Mclust Gaussian Mixture Model into a Python 
    dictionary.
    """

    def __init__(self, r_model, predictions = None):
        """The Constructor"""
        self.rmodel         = r_model
        self.model          = _listvector_to_dict(r_model)
        self.classification = _listvector_to_dict(predictions) if predictions is not None else None
        self._predicted     = False if predictions is None else True
        self._manage_parameters()

    def __repr__(self):
        return "GaussianMixtureModel()"
    
    def __str__(self):
        str_ = \
f"""----------------------------------------------------
Python wrapper for R Mclust's Gaussian Mixture Model
            fitted by EM algorithm
----------------------------------------------------
.rmodel : R model object as rpy2.robjects
.model  : R Model translation into Py dict

Mclust {self.model['modelName']} model fitted with {self.ndG[2]} components:
-----------------------------------------------------
log-likelihood : {self.loglik}
n : {self.ndG[0]}
df : {self.ndG[1]}
BIC : {self.bic['BIC']}
ICL : {self.model['icl']}

Predicted : {self._predicted}

"""
        return str_

    @property
    def data(self):
        """
        The data used to fit the model.
        """
        return _np.reshape(self.model['data'], (self.ndG[1], self.ndG[0])).T
 

    @property
    def bic(self):
        """
        The Bayesian Information Criterion of the model.
        """
        return {"bic": self.model["bic"], "BIC": self.model["BIC"]}

    @property
    def ndG(self):
        """
        Array containing in order:
        - n : the number of data points used to fit the model
        - d : the number of parameters/features fitted with the model
        - G : the number of components in the model
        """
        return _np.array([self.model["n"], self.model["d"], self.model["G"]])

    @property
    def loglik(self):
        """
        The log-likelihood of the model.
        """
        return self.model["loglik"]

    @property
    def train_classification(self):
        """
        Array containing in order:
        - z : the membership probability of each data point to each component
        - classification : the classification of the data points
        """
        return {'z': self.model["z"], 'classification': self.model["classification"]}

    @property
    def parameters(self):
        """
        Dictionary containint all the parameters estimated by the model, for each component
        Components:
        - pro : the mixing proportions
        - mean : the means
        - variance : another dictionary containing the variance and covariance matrices:
        - covMats : the covariance matrices for each group of the model
        - sigmasq : the variance of the features of the model
        """
        return self.model["parameters"]

    @property
    def uncertainty(self):
        """
        The uncertainty of the model for each data point membership probability.
        """
        return self.model["uncertainty"]

    @property
    def coeffs(self):
        """
        The coefficients of the model.
        """
        return self.model['parameters']["mean"]

    def predict(self, data):
        """
        Predict the membership probability of each data point to each component.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            The data to predict the membership probability for.
        
        Returns
        -------
        array-like, shape (n_samples, n_components)
            The predicted membership probability for each data point.
        """
        import os
        from .r_check import check_packages
        from grasp.core.folder_paths import R_SOURCE_FOLDER as _RSF
        check_packages("mclust")
        _np2r.activate()
        code = os.path.join(_RSF, "gaussian_mixture.R")
        _R(f'source("{code}")')
        rdata = _np2r.py2rpy(data)
        result = _genv['GMMPredict'](self.rmodel, rdata)
        self.classification = _listvector_to_dict(result)
        self._predicted = True
        return self.classification
    
    def save_model(self, filename):
        """
        Save the R model to a `.rds` file, so it can be loaded both into R, with
        `model <- readRDS(filename)`, and in python, with 
        
        ```python
        model = grasp.RegressionModel()
        model.load_model(filename)
        ```
        
        Parameters
        ----------
        filename : str
            The name of the file to save the model to. The extention can be
            omitted, as it will be attached to then filename
        """
        if '.rds' not in filename:
            filename += '.rds'
        _ro.r('saveRDS')(self.rmodel, file=filename)
        print(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, filename):
        """
        Load the model from a `.rds` file, so it can be used in Python.
        
        Parameters
        ----------
        filename : str
            The name of the file to load the model from. The extention can be
            omitted, as it will be attached to then filename
        """
        if '.rds' not in filename:
            filename += '.rds'
        rmodel = _ro.r('readRDS')(filename)
        return cls(r_model=rmodel)


    def _manage_parameters(self):
        sigma = self.model["parameters"]["variance"]["sigma"]
        # Handle different shapes of sigma
        self.model["parameters"]["variance"]["covMats"] = _np.reshape(sigma, (self.ndG[2], self.ndG[1], self.ndG[1]))
        self.model["parameters"]["mean"] = _np.reshape(_np.array(self.model["parameters"]["mean"]), (self.ndG[2], self.ndG[1]))
        keys_to_remove = ["modelName", "d", "G", "sigma", "scale", 'shape']
        for key in keys_to_remove:
            self.model["parameters"]['variance'].pop(key, None)



class KFoldGMM:
    """
    Python wrapper for the result of K-fold cross-validation of a Gaussian Mixture Model (Mclust) in R.
    """
    def __init__(self, rmodel: object):
        """
        Parameters
        ----------
        r_result : rpy2.robjects.ListVector or dict
            The result returned by the R KFoldGMM function.
        """
        self.rmodel = rmodel
        self.model = _listvector_to_dict(rmodel)

    def __repr__(self) -> str:
        return f"KFoldGMM_Cross-Validation(K={self.n_folds}, mean_loglik={self.mean_loglik:.3f}, mean_bic={self.mean_bic:.3f})"

    def __str__(self) -> str:
        return (
            f"K-Fold GMM Cross-Validation Result\n"
            f"-----------------------------------\n"
            f"Folds: {self.n_folds}\n"
            f"Mean Log-Likelihood: {self.mean_loglik:.3f}\n"
            f"Mean BIC: {self.mean_bic:.3f}\n"
            f"Log-Likelihoods: {self.logliks}\n"
            f"BICs: {self.bics}\n"
        )

    @property
    def mean_loglik(self) -> float:
        """
        The mean log-likelihood across all folds.
        """
        return float(self.model.get("mean_loglik", float('nan')))

    @property
    def mean_bic(self) -> float:
        """
        The mean BIC across all folds.
        """
        return float(self.model.get("mean_bic", float('nan')))

    @property
    def logliks(self) -> list:
        """
        List of log-likelihoods for each fold.
        """
        return list(self.model.get("logliks", []))

    @property
    def bics(self) -> list:
        """
        List of BIC values for each fold.
        """
        return list(self.model.get("bics", []))

    @property
    def n_folds(self) -> int:
        """
        Number of folds used in cross-validation.
        """
        return len(self.logliks)

    @property
    def models(self) -> list[GaussianMixtureModel]:
        """
        List of fitted GMM models for each fold, as GaussianMixtureModel instances.
        """
        r_models = self.rmodel[4]
        # If r_models is a single model, wrap in list
        model_list = [GaussianMixtureModel(r_model) for r_model in r_models]
        return model_list
    
    
    def best_model(self) -> GaussianMixtureModel:
        """
        Returns the best model based on BIC.
        """
        best_index = self.bics.index(min(self.bics))
        return self.models[best_index]



class RegressionModel:
    """
    Class to convert the R LM Regression Model into a Python dictionary.
    """

    def __init__(self, r_model= None, kind:str=''):
        """The Constructor"""
        if not r_model is None:
            self.rmodel = r_model
            self.model  = _listvector_to_dict(r_model)
            self._model_kind = kind if kind!='' else self.model['kind']
        else:
            self.rmodel = None
            self.model = None
            self._model_kind = kind


    def __repr__(self):
        """The representation of the model."""
        return f"RegressionModel('{self._model_kind.capitalize()}')"
    

    def __str__(self):
        """The string representation of the model."""
        if self.model is None:
            return ""
        txt = _kde_labels(self._model_kind, self.coeffs).splitlines()
        txt.pop(0)
        txt = ('\n'.join(txt)).replace('$', '')
        str_ = \
f"""----------------------------------------------------
Python wrapper for R Levenberg-Marquardt Nonlinear 
              Least-Squares Algorithm            
----------------------------------------------------
.rmodel : R model object as rpy2.robjects
.model  : R Model translation into Py dict

{self._model_kind.upper()} Regression Model
----------------------------------------------------
{txt}
"""
        return str_
    
    def save_model(self, filename):
        """
        Save the R model to a `.rds` file, so it can be loaded both into R, with
        `model <- readRDS(filename)`, and in python, with 
        
        ```python
        model = grasp.RegressionModel()
        model.load_model(filename)
        ```
        
        Parameters
        ----------
        filename : str
            The name of the file to save the model to. The extention can be
            omitted, as it will be attached to then filename
        """
        if '.rds' not in filename:
            filename += '.rds'
        _ro.r('saveRDS')(self.rmodel, file=filename)
        print(f"Model saved to {filename}")


    @classmethod
    def load_model(cls, filename):
        """
        Load the model from a `.rds` file, so it can be used in Python.
        
        Parameters
        ----------
        filename : str
            The name of the file to load the model from. The extention can be
            omitted, as it will be attached to then filename
        """
        if '.rds' not in filename:
            filename += '.rds'
        rmodel = _ro.r('readRDS')(filename)
        return cls(r_model=rmodel)


    @property
    def coeffs(self):
        """
        The coefficients of the regression model.
        """
        return self.model["coeffs"]
    
    @property
    def data(self):
        """
        The data used to fit the model.
        """
        return self.model["data"]
    
    @property
    def x(self):
        """
        The independent variables of the model.
        """
        return self.model["x"]

    @property
    def y(self):
        """
        The dependent variables of the model.
        """
        return self.model["y"]
    
    @property
    def residuals(self):
        """
        The residuals of the model.
        """
        return self.model["residuals"]

    @property
    def kind(self):
        """
        The kind of model fitted to the data.
        """
        return self._model_kind


class PyRegressionModel:
    """
    Native Python regression model class for the GRASP package.
    """
    def __init__(self, fit, kind):
        """The Constructor"""
        if kind == 'linear':
            self.data = {
                "x": fit["x"],
                "y": fit["data"],
            }
        else:
            self.data = fit['data']
        self.x = fit["x"]
        self.y = fit["y_fit"]
        self.residuals = fit["residuals"]
        self.kind = kind
        self.coeffs = fit["parameters"]
        self.covariance_matrix = fit["covmat"]

    def __repr__(self):
        """The representation of the model."""
        return f"RegressionModel('{self.kind.capitalize()}')"
    
    def __str__(self):
        """The string representation of the model.""" 
        base_txt = f"""{self.kind.upper()} Regression Model
--------------------------------------
Coefficients:
"""
        coeff_txt = "\n".join([f"{chr(65 + i)} = {_format_number(param)}" for i, param in enumerate(self.coeffs)])
        return base_txt + coeff_txt


# =============================================================================
# Support scripts for other functionalities related to R2Py Models
# =============================================================================

def _listvector_to_dict(r_listvector):
    """
    Recursively converts an R ListVector (from rpy2) to a nested Python dictionary.
    """
    py_dict = {}
    for key, value in r_listvector.items():
        # Handle simple types
        if isinstance(
            value,
            (
                _ro.rinterface.IntSexpVector,
                _ro.rinterface.FloatSexpVector,
                _ro.rinterface.StrSexpVector,
            ),
        ):
            py_dict[key] = list(value) if len(value) > 1 else value[0]
        # Handle data frames (convert to pandas DataFrame)
        elif isinstance(value, _ro.vectors.DataFrame):
            py_dict[key] = _pd2r.rpy2py(value)
        # Handle nested ListVectors
        elif isinstance(value, _ro.vectors.ListVector):
            py_dict[key] = _listvector_to_dict(value)
        # Handle other lists
        elif isinstance(value, _ro.vectors.Vector):
            py_dict[key] = list(value)
        # Other R objects could be added as necessary
        else:
            py_dict[key] = value
    return py_dict


def _format_number(num):
    """
    Format the number using scientific notation if it is too large or too
    small.
    """
    if abs(num) < 1e-3 or abs(num) > 1e3:
        return f"{num:.2e}"
    else:
        return f"{num:.3f}"


def _kde_labels(kind: str, coeffs):
    """
    Return the labels for the KDE plot.
    """
    
    if kind == 'gaussian':
        A, mu, sigma2 = coeffs
        label = f"""Gaussian
$A$   = {_format_number(A)}
$\\mu$   = {_format_number(mu)}
$\\sigma^2$  = {_format_number(sigma2)}"""
        
    elif kind == 'boltzmann':
        A1, A2, x0, dx = coeffs
        label = f"""Boltzmann
$A1$   = {_format_number(A1)}
$A2$   = {_format_number(A2)}
$x_0$   = {_format_number(x0)}
$dx$   = {_format_number(dx)}"""
        
    elif kind == 'exponential':
        A, lmbda = coeffs
        label = f"""Exponential
$A$   = {_format_number(A)}
$\\lambda$ = {_format_number(lmbda)}"""
        
    elif kind == 'king':
        A, ve, sigma = coeffs
        label = f"""King
$A$   = {_format_number(A)}
$v_e$   = {_format_number(ve)}
$\\sigma$  = {_format_number(sigma)}"""
        
    elif kind == 'maxwell':
        A, sigma = coeffs
        label = f"""Maxwell
$A$   = {_format_number(A)}
$\\sigma$  = {_format_number(sigma)}"""
        
    elif kind == 'rayleigh':
        A, sigma = coeffs
        label = f"""Rayleigh
$A$   = {_format_number(A)}
$\\sigma$  = {_format_number(sigma)}"""
        
    elif kind == 'lorentzian':
        A, x0, gamma = coeffs
        label = f"""Lorentzian
$A$   = {_format_number(A)}
$x_0$   = {_format_number(x0)}
$\\gamma$  = {_format_number(gamma)}"""
    
    elif kind == 'power':
        A, n = coeffs
        label = f"""Power
$A$   = {_format_number(A)}
$n$   = {_format_number(n)}"""
        
    elif kind == 'lognormal':
        A, mu, sigma = coeffs
        label = f"""Lognormal
$A$   = {_format_number(A)}
$\mu$   = {_format_number(mu)}
$\\sigma$  = {_format_number(sigma)}"""
    
    elif kind == 'poisson':
        A, lmbda = coeffs
        label = f"""Poisson
$A$   = {_format_number(A)}
$\\lambda$ = {_format_number(lmbda)}"""
        
    elif kind == 'linear':
        A, B = coeffs
        label = f"""Linear
$A$   = {_format_number(A)}
$B$   = {_format_number(B)}"""
    
    elif 'polynomial' in kind:
        degree = int(kind[0])
        label = f"Polynomial of degree {degree}\n"+\
        "\n".join(
            [f"${chr(65 + i)}$   = {_format_number(param)}" for i, param in enumerate(coeffs)]
        )

    elif kind == 'custom':
        label = "Custom\n" + "\n".join(
            [f"${chr(65 + i)}$   = {_format_number(param)}" for i, param in enumerate(coeffs)]
        )
        
    return label


