"""
Author(s)
---------
    - Pietro Ferraiuolo : Written in 2024

Description
-----------
Base class for formulas calsses, used in the 'grasp.functions' module.
"""

from abc import ABC
import numpy as _np
from sympy import Basic as _sb
from typing import List as _List
from numpy.typing import ArrayLike as _ArrayLike


class BaseFormula(ABC):
    """
    Base class for the various formula calsses
    """

    def __init__(self):
        """The constructor"""
        self._name: str = None
        self._variables: _List[_sb] = []
        self._formula: _sb = None
        self._errFormula: _sb = None
        self._errVariables: _List[_sb] = None
        self._correlations: _List[_sb] = None
        self._values: _ArrayLike = None
        self._errors: _ArrayLike = None

    def __str__(self) -> str:
        """String representation"""
        return self._get_str()

    def __repr__(self) -> str:
        """Return the analytical formula as a string"""
        return self._get_str()

    def _get_str(self):
        """Return the actual formula as a string"""
        formula = self._formula
        computed = (True if self._values is not None else False) or (
            True if self._errors is not None else False
        )
        return f"{self._name}:\n{formula}\nComputed: {computed}"

    @property
    def formula(self) -> _sb:
        """Return the formula"""
        return self._formula

    @property
    def variables(self) -> _List[_sb]:
        """Return the variables"""
        return self._variables

    @property
    def computed_values(self) -> _ArrayLike:
        """Return the values"""
        return "Not computed" if self._values is None else self._values

    @property
    def computed_errors(self) -> _ArrayLike:
        """Return the errors"""
        return "Not computed" if self._errors is None else self._errors

    @property
    def error_formula(self) -> _sb:
        """Return the error formula"""
        return self._errFormula

    @property
    def error_variables(self) -> _List[_sb]:
        """Return the error variables"""
        return self._errVariables

    @property
    def correlations(self) -> _List[_sb]:
        """Return the correlations"""
        return self._correlations

    def compute(
        self,
        data: _List[_ArrayLike],
        errors: _List[_ArrayLike] = None,
        correlations: _List[_ArrayLike] = None,
    ) -> _ArrayLike:
        """
        Compute the values of the formula, along with the propagated errors if
        variables errors (and correlation) are provided.

        Parameters
        ----------
        data : List[ArrayLike]
            The data to use for the computation.
        errors: List[ArrayLike], optional
            If provided, the propagated errors will be computed.
        correlations: List[ArrayLike], optional
            If provided, the correlations will be used in the error computation.

        Returns
        -------
        result : ArrayLike
            The computed values.
        """
        from grasp.analyzers.calculus import compute_numerical_function
        print(
            f"""WARNING! Be sure that the input data follow this specific order: 
Data:         {self.variables}"""
        )
        if errors is not None:
            from grasp.analyzers.calculus import compute_error
            print(
                f"""Errors:       {self._errVariables}
Correlations: {self._correlations}
"""
                + "-" * 30
            )
            variables = self._variables + self._errVariables
            print("Errors:")
            if self._correlations is None:
                self._errors = compute_error(self._errFormula, variables, data, errors)
            else:
                variables += self._correlations
                if correlations is None:
                    correlations = [
                        _np.zeros(len(data[0])) for _ in range(len(self._correlations))
                    ]
                self._errors = compute_error(
                    self._errFormula, variables, data, errors, corr_values=correlations
                )
        print("Data:")
        self._values = compute_numerical_function(self._formula, self._variables, data)
        return self

    # @abstractmethod
    # def _get_formula(self) -> _sb:
    #     """Return the derived formula of the class"""
    #     pass

    # @abstractmethod
    # def _analitical_formula(self) -> str:
    #     """Return the analitical formula for the class"""
    #     pass
