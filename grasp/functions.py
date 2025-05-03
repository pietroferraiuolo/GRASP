"""
Module: functions.py

Author(s)
---------
- Pietro Ferraiuolo : Written in 2024

Description
-----------
This module provides a set of classes to compute various astronomical and
physical quantities such as angular separation, line-of-sight distance, radial
distances in 2D and 3D, total velocity, effective gravitational potential, and
cartesian conversion.

Classes
-------
- AngularSeparation(ra0, dec0)
    Computes the angular separation between two points in the sky.

- LosDistance()
    Computes the line-of-sight distance based on parallax.

- RadialDistance2D(gc_distance)
    Computes the 2D-projected radial distance of a source from the center of a cluster or given RA/DEC coordinates.

- RadialDistance3D(gc_distance=None)
    Computes the 3D radial distance of a source from the center of a cluster or given RA/DEC coordinates.

- TotalVelocity()
    Computes the total velocity based on the given velocity components.

- EffectivePotential(shell=False)
    Computes the effective gravitational potential, optionally considering a shell model.

- CartesianConversion(ra0=0, dec0=0)
    Computes the cartesian conversion of coordinates and velocities.

How to Use
----------
1. Import the module:
    ```python
    from grasp import functions
    ```

2. Create an instance of the desired class and call the appropriate methods:
    ```python
    angular_sep = functions.AngularSeparation(ra0=10.0, dec0=20.0)
    # Say we have some data
    data = [ra, dec]
    result = angular_sep.compute(data)
    ```

Examples
--------
Example usage of `AngularSeparation` class:
    ```python
    from grasp import functions
    from astropy import units as u

    ra0 = 10.0 * u.deg
    dec0 = 20.0 * u.deg
    angular_sep = functions.AngularSeparation(ra0, dec0)
    print(angular_sep)
    ```

Example usage of `LosDistance` class:
    ```python
    from grasp import functions

    los_dist = functions.LosDistance()
    print(los_dist)
    ```

Example usage of `RadialDistance2D` class:
    ```python
    from grasp import functions
    from astropy import units as u

    gc_distance = 1000 * u.pc
    radial_dist_2d = functions.RadialDistance2D(gc_distance)
    print(radial_dist_2d)
    ```

Example usage of `RadialDistance3D` class:
    ```python
    from grasp import functions
    from astropy import units as u

    gc_distance = 1000 * u.pc
    radial_dist_3d = functions.RadialDistance3D(gc_distance=gc_distance)
    print(radial_dist_3d)
    ```

Example usage of `TotalVelocity` class:
    ```python
    from grasp import functions

    total_vel = functions.TotalVelocity()
    print(total_vel)
    ```

Example usage of `EffectivePotential` class:
    ```python
    from grasp import functions

    eff_pot = functions.EffectivePotential(shell=True)
    print(eff_pot)
    ```

Example usage of `CartesianConversion` class:
    ```python
    from grasp import functions

    cart_conv = functions.CartesianConversion(ra0=10.0, dec0=20.0)
    print(cart_conv)
    ```
"""

import sympy as _sp
from astropy.table import Table as _Table
from numpy.typing import ArrayLike as _ArrayLike
from typing import List as _List
from grasp.analyzers.calculus import (
    compute_numerical_function as _compute_numerical,
    error_propagation as _error_propagation,
)


class CartesianConversion:
    r"""
    Class for the analytical cartesian conversion.

    The cartesian conversion is defined as

    :math:`x = \sin(\alpha - \alpha_0) \cos(\delta_0)`

    :math:`y = \sin(\delta)\cos(\delta_0) - \cos(\delta)\sin(\delta_0)\cos(\alpha - \alpha_0)`

    :math:`r = \sqrt{x^2 + y^2}`.
    """

    def __init__(self, ra0 = None, dec0 = None, propagate_error: bool = True):
        """The constructor"""
        super().__init__()
        self._values = None
        self._formula = None
        self._variables = None
        self._errFormula = None
        self._errVariables = None
        self._corVariables = None
        self.ra0 = ra0 if ra0 is not None else _sp.symbols("alpha_0")
        self.dec0 = dec0 if dec0 is not None else _sp.symbols("delta_0")
        self._get_formula(propagate_error)

    @property
    def x(self):
        """
        Return the x component of the cartesian conversion.
        """
        if self._values is None or self._values[0] is None:
            res = self._formula[0]
        else:
            res = self._values[0]
        return res

    @property
    def x_error(self):
        """
        Return the error of the x component of the cartesian conversion.
        """
        if self._errFormula is None or self._errFormula[0].get("error_formula") is None:
            res = self._formula[0]
        else:
            res = self._errFormula[0]["error_formula"]
        return res

    @property
    def y(self):
        """
        Return the y component of the cartesian conversion.
        """
        if self._values is None or self._values[1] is None:
            res = self._formula[1]
        else:
            res = self._values[1]
        return res

    @property
    def y_error(self):
        """
        Return the error of the y component of the cartesian conversion.
        """
        if self._errFormula is None or self._errFormula[1].get("error_formula") is None:
            res = self._formula[1]
        else:
            res = self._errFormula[1]["error_formula"]
        return res

    @property
    def r(self):
        """
        Return the r component of the cartesian conversion.
        """
        if self._values is None or self._values[2] is None:
            res = self._formula[2]
        else:
            res = self._values[2]
        return res

    @property
    def r_error(self):
        """
        Return the error of the r component of the cartesian conversion.
        """
        if self._errFormula is None or self._errFormula[2].get("error_formula") is None:
            res = self._formula[2]
        else:
            res = self._errFormula[2]["error_formula"]
        return res

    @property
    def theta(self):
        """
        Return the theta component of the cartesian conversion.
        """
        if self._values is None or self._values[3] is None:
            res = self._formula[3]
        else:
            res = self._values[3]
        return res

    @property
    def theta_error(self):
        """
        Return the error of the theta component of the cartesian conversion.
        """
        if self._errFormula is None or self._errFormula[3].get("error_formula") is None:
            res = self._formula[3]
        else:
            res = self._errFormula[3]["error_formula"]
        return res

    @property
    def mu_x(self):
        """
        Return the mu_x component of the cartesian conversion.
        """
        if self._values is None or self._values[4] is None:
            res = self._formula[4]
        else:
            res = self._values[4]
        return res

    @property
    def mux_error(self):
        """
        Return the error of the mu_x component of the cartesian conversion.
        """
        if self._errFormula is None or self._errFormula[4].get("error_formula") is None:
            res = self._formula[4]
        else:
            res = self._errFormula[4]["error_formula"]
        return res

    @property
    def mu_y(self):
        """
        Return the mu_y component of the cartesian conversion.
        """
        if self._values is None or self._values[5] is None:
            res = self._formula[5]
        else:
            res = self._values[5]
        return res

    @property
    def muy_error(self):
        """
        Return the error of the mu_y component of the cartesian conversion.
        """
        if self._errFormula is None or self._errFormula[5].get("error_formula") is None:
            res = self._formula[5]
        else:
            res = self._errFormula[5]["error_formula"]
        return res

    @property
    def mu_r(self):
        """
        Return the mu_r component of the cartesian conversion.
        """
        if self._values is None or self._values[6] is None:
            res = self._formula[6]
        else:
            res = self._values[6]
        return res

    @property
    def mur_error(self):
        """
        Return the error of the mu_r component of the cartesian conversion.
        """
        if self._errFormula is None or self._errFormula[6].get("error_formula") is None:
            res = self._formula[6]
        else:
            res = self._errFormula[6]["error_formula"]
        return res

    @property
    def mu_theta(self):
        """
        Return the mu_theta component of the cartesian conversion.
        """
        if self._values is None or self._values[7] is None:
            res = self._formula[7]
        else:
            res = self._values[7]
        return res

    @property
    def mutheta_error(self):
        """
        Return the error of the mu_theta component of the cartesian conversion.
        """
        if self._errFormula is None or self._errFormula[7].get("error_formula") is None:
            res = self._formula[7]
        else:
            res = self._errFormula[7]["error_formula"]
        return res


    def _get_formula(self, propagate_error: bool = True):
        """Analytical formula getter for the cartesian conversion"""
        ra, dec = _sp.symbols("alpha delta")
        pmra, pmdec = _sp.symbols("mu_alpha mu_delta")
        variables = [ra, dec, pmra, pmdec]
        # cartesian spatial coordinates
        x = _sp.sin(ra - self.ra0) * _sp.cos(self.dec0)
        y = _sp.sin(dec) * _sp.cos(self.dec0) - _sp.cos(dec) * _sp.sin(
            self.dec0
        ) * _sp.cos(ra - self.ra0)
        # polar spatial coordinates
        r = _sp.sqrt(x**2 + y**2)
        theta = _sp.atan2(x, y)
        # cartesian velocity components
        mu_x = pmra * _sp.cos(ra - self.ra0) - pmdec * _sp.sin(dec) * _sp.sin(
            ra - self.ra0
        )
        mu_y = pmra * _sp.sin(self.dec0) * _sp.sin(ra - self.ra0) + pmdec * (
            _sp.cos(dec) * _sp.cos(self.dec0)
            + _sp.sin(dec) * _sp.sin(self.dec0) * _sp.cos(ra - self.ra0)
        )
        # polar velocity components
        mu_r = (x * mu_x + y * mu_y) / r
        mu_theta = (y * mu_x - x * mu_y) / (r**2)
        self._formula = [x, y, r, theta, mu_x, mu_y, mu_r, mu_theta]
        self._variables = variables
        if propagate_error:
            self.error_propagation()
        return self
    
    def error_propagation(self):
        """
        Compute the error propagation for the cartesian conversion.

        Returns
        -------
        self
            The cartesian conversion errors, stored in the .errFormula method as
            a pandas DataFrame.
        """
        ra, dec = _sp.symbols("alpha delta")
        pmra, pmdec = _sp.symbols("mu_alpha mu_delta")
        x, y, r, theta, mu_x, mu_y, mu_r, mu_theta = self._formula
        xerr = _error_propagation(x, [ra, dec], correlation=True)
        yerr = _error_propagation(y, [ra, dec], correlation=True)
        rerr = _error_propagation(r, [ra, dec], correlation=True)
        thetaerr = _error_propagation(theta, [ra, dec], correlation=True)
        muxerr = _error_propagation(mu_x, [ra, dec, pmra, pmdec], correlation=True)
        muyerr = _error_propagation(mu_y, [ra, dec, pmra, pmdec], correlation=True)
        murerr = _error_propagation(mu_r, [ra, dec, pmra, pmdec], correlation=True)
        muthetaerr = _error_propagation(
            mu_theta, [ra, dec, pmra, pmdec], correlation=True
        )
        self._errFormula = [
            xerr,
            yerr,
            rerr,
            thetaerr,
            muxerr,
            muyerr,
            murerr,
            muthetaerr,
        ]
        self._errVariables = (
            rerr["error_variables"]["errors"] + murerr["error_variables"]["errors"]
        )
        self._corVariables = (
            rerr["error_variables"]["corrs"] + murerr["error_variables"]["corrs"]
        )

    def compute(
        self,
        data: _List[_ArrayLike],
        errors: _List[_ArrayLike] = None,
        correlations: _List[_ArrayLike] = None,
    ):
        """
        Compute the cartesian conversion.

        Parameters
        ----------
        data :_List[ArrayLike]
            The data to use for the computation. Needs to be in the order
            [ra, dec, pmra, pmdec].
        errors :_List[ArrayLike], optional
            The errors to use for the computation. Needs to be in the order
            [ra_err, dec_err, pmra_err, pmdec_err].
        correlations :_List[ArrayLike], optional
            The correlations to use for the computation. Needs to be in the order 
            [ra_dec_corr, ra_pmra_corr, ra_pmdec_corr,dec_pmra_corr, dec_pmdec_corr,
            pmra_pmdec_corr].

        Returns
        -------
        self
            The cartesian conversion computed quantities, stored in the .values method as
            a pandas DataFrame.
        """
        quantities = [
            self.x,
            self.y,
            self.r,
            self.theta,
            self.mu_x,
            self.mu_y,
            self.mu_r,
            self.mu_theta,
        ]
        tags = ["x", "y", "r", "theta", "mu_x", "mu_y", "mu_r", "mu_theta"]
        var_set = [self._variables[:2]] * 4 + [self._variables] * 4
        q_values = _Table()
        if len(data) == 2:
            for var, eq, name in zip(var_set[:4], quantities, tags):
                result = _compute_numerical(eq, var, data)
                q_values[name] = result
                self.compute_error(data, errors, correlations)
        else:
            for var, eq, name in zip(var_set, quantities, tags):
                result = _compute_numerical(eq, var, data)
                q_values[name] = result
            if errors is not None:
                self.compute_error(data, errors, correlations)
        self._values = q_values
        return self
