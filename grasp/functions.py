"""Symbolic helpers for sky-plane geometry.

Currently this module exposes a single class:

- :class:`CartesianConversion` -- tangent-plane (gnomonic) projection of
  Gaia-style ``(ra, dec, pmra, pmdec)`` data around a reference point
  ``(ra0, dec0)``, with optional analytical error propagation.

The trigonometric expressions internally rely on
:mod:`sympy` and therefore expect *radians* — by default
:class:`CartesianConversion` accepts angles in degrees and converts them to
radians for the symbolic backend (see the ``input_unit`` parameter).
"""

import math as _math
from typing import Literal as _Literal

import sympy as _sp
from astropy import units as _u
from astropy.table import Table as _Table
from numpy.typing import ArrayLike as _ArrayLike

from grasp.analyzers.calculus import (
    compute_numerical_function as _compute_numerical,
)
from grasp.analyzers.calculus import (
    error_propagation as _error_propagation,
)


def _to_radians(value, input_unit: str):
    """Convert an angle to radians for sympy consumption.

    Parameters
    ----------
    value : float, int, astropy.units.Quantity, sympy.Expr or None
        The angle to convert. ``None`` is returned unchanged.
    input_unit : {"deg", "rad"}
        Unit assumed for plain numbers/sympy expressions. Quantities are
        always handled via :mod:`astropy.units`.
    """
    if value is None:
        return None
    if isinstance(value, _u.Quantity):
        return float(value.to(_u.rad).value)
    if isinstance(value, _sp.Basic):
        if input_unit == "deg":
            return value * _sp.pi / 180
        return value
    if input_unit == "deg":
        return float(value) * _math.pi / 180.0
    if input_unit == "rad":
        return float(value)
    raise ValueError(
        f"input_unit must be 'deg' or 'rad', got {input_unit!r}"
    )


class CartesianConversion:
    r"""Tangent-plane (gnomonic) projection of Gaia data.

    The conversion is

    .. math::

        x = \sin(\alpha - \alpha_0) \cos(\delta_0)

        y = \sin(\delta)\cos(\delta_0) - \cos(\delta)\sin(\delta_0)\cos(\alpha - \alpha_0)

        r = \sqrt{x^2 + y^2}

    The :mod:`sympy` trig functions expect **radians**. Gaia angles are
    distributed in degrees; this class therefore accepts angles in
    degrees by default and converts them to radians internally
    (``input_unit="deg"``). Pass ``input_unit="rad"`` if you have already
    converted, or supply :class:`astropy.units.Quantity` values, which are
    always converted via ``.to(u.rad)``.

    Parameters
    ----------
    ra0, dec0 : float, int, :class:`~astropy.units.Quantity`, sympy.Expr or None
        Reference right ascension and declination. If ``None``, sympy
        symbols ``alpha_0`` and ``delta_0`` are used (purely analytical
        mode).
    propagate_error : bool, optional
        If ``True`` (default), the analytical error propagation is
        computed and stored on ``self._errFormula``.
    input_unit : {"deg", "rad"}, optional
        Unit assumed for ``ra0``/``dec0`` and for any plain-number entries
        in :meth:`compute`'s ``data`` argument. Default is ``"deg"``.

    See Also
    --------
    astropy.coordinates.SkyCoord.separation : reference implementation we
        validate against in the unit tests.
    """

    def __init__(
        self,
        ra0=None,
        dec0=None,
        propagate_error: bool = True,
        input_unit: _Literal["deg", "rad"] = "deg",
    ):
        """The constructor"""
        super().__init__()
        if input_unit not in {"deg", "rad"}:
            raise ValueError(
                f"input_unit must be 'deg' or 'rad', got {input_unit!r}"
            )
        self._values = None
        self._formula = None
        self._variables = None
        self._errFormula = None
        self._errVariables = None
        self._corVariables = None
        self.input_unit = input_unit
        self.ra0 = (
            _to_radians(ra0, input_unit)
            if ra0 is not None
            else _sp.symbols("alpha_0")
        )
        self.dec0 = (
            _to_radians(dec0, input_unit)
            if dec0 is not None
            else _sp.symbols("delta_0")
        )
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
        data: list[_ArrayLike],
        errors: list[_ArrayLike] = None,
        correlations: list[_ArrayLike] = None,
    ):
        """
        Compute the cartesian conversion.

        Parameters
        ----------
        data : _List[ArrayLike]
            The data to use for the computation. Order is ``[ra, dec]`` or
            ``[ra, dec, pmra, pmdec]``. Angles are expected in the unit set
            by ``input_unit`` at construction time (defaults to degrees).
            :class:`~astropy.units.Quantity` arrays are accepted as well
            and will be converted to radians via ``.to(u.rad)``.
        errors : _List[ArrayLike], optional
            Errors corresponding to ``data``. Angular errors are converted
            using the same convention as ``data``.
        correlations : _List[ArrayLike], optional
            Correlation arrays, in the order ``[ra_dec, ra_pmra, ra_pmdec,
            dec_pmra, dec_pmdec, pmra_pmdec]``. Correlations are
            dimensionless and are passed through unchanged.

        Returns
        -------
        self
            The cartesian conversion computed quantities, stored in the .values method as
            a pandas DataFrame.
        """
        data = self._convert_input(data)
        if errors is not None:
            errors = self._convert_input(errors)
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
            # Only positions are supplied: compute the spatial quantities and
            # skip the proper-motion branch entirely.
            for var, eq, name in zip(var_set[:4], quantities[:4], tags[:4], strict=False):
                result = _compute_numerical(eq, var, data)
                q_values[name] = result
        else:
            for var, eq, name in zip(var_set, quantities, tags, strict=False):
                result = _compute_numerical(eq, var, data)
                q_values[name] = result
        if errors is not None:
            # The pre-existing implementation did not actually provide a
            # standalone ``compute_error`` numerical helper. Errors stay
            # analytical (``self._errFormula``); numerical evaluation of the
            # propagated error is left to ``BaseFormula``-backed wrappers.
            _ = correlations  # kept for API compatibility
        self._values = q_values
        return self

    def _convert_input(self, data: list[_ArrayLike]) -> list[_ArrayLike]:
        """Convert angular entries to radians for symbolic evaluation.

        The first two columns of ``data`` are ``ra``/``dec`` and must be
        in radians for SymPy; subsequent columns are proper motions and
        are passed through unchanged.
        """
        import numpy as _np

        converted = []
        for idx, column in enumerate(data):
            if column is None or idx >= 2:
                converted.append(column)
                continue
            if isinstance(column, _u.Quantity):
                converted.append(column.to(_u.rad).value)
            elif self.input_unit == "deg":
                converted.append(_np.asarray(column, dtype=float) * (_math.pi / 180.0))
            else:
                converted.append(column)
        return converted
