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
from grasp import types as _gt
from astropy import units as _u
_QTable = _gt.QTable
_ArrayLike = _gt.ArrayLike


class BaseFormula(ABC):
    """
    Base class for the various formula calsses
    """

    def __init__(self):
        """The constructor"""
        self._name: str = None
        self._variables: list[_sb] = []
        self._formula: _sb = None
        self._errFormula: _sb = None
        self._errVariables: list[_sb] = None
        self._correlations: list[_sb] = None
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
    def variables(self) -> list[_sb]:
        """Return the variables"""
        return self._variables

    @property
    def computed_values(self) -> _ArrayLike:
        """Return the values"""
        return "Not computed" if self._values is None else self._values

    values: _ArrayLike = computed_values

    @property
    def computed_errors(self) -> _ArrayLike:
        """Return the errors"""
        return "Not computed" if self._errors is None else self._errors

    errors: _ArrayLike = computed_errors

    @property
    def error_formula(self) -> _sb:
        """Return the error formula"""
        return self._errFormula

    @property
    def error_variables(self) -> list[_sb]:
        """Return the error variables"""
        return self._errVariables

    @property
    def correlations(self) -> list[_sb]:
        """Return the correlations"""
        return self._correlations

    def compute(
        self,
        data: list[_ArrayLike],
        errors: list[_ArrayLike] = None,
        correlations: list[_ArrayLike] = None,
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
        self._values = compute_numerical_function(self._formula, self._variables, data)
        return self


class BaseSample(_QTable):
    
    """
    Base class for Sample and GcSample.
    It is used to define the common methods and attributes for both classes.
    """

    save = _QTable.write
    header = _QTable.meta

    def __init__(
        self,
        data: _gt.AstroTable = None,
        **kwargs: dict[str, _gt.Any],
    ):
        """
        Class for handling the query result sample.

        It is an object with unifies the Cluster object and the sample data obtained,
        in order to have a compact object which has everythin and from which quantities
        can be computed easily.

        Parameters
        ----------
        sample : astropy.Table, pandas.DataFrame, dict
            Table containing the retrieved sample's data.
        gc : grasp.cluster.Cluster, optional
            Globular cluster object used for the query.
        """
        # Accepts data as QTable, Table, DataFrame, or dict
        if isinstance(data, _gt.DataFrame):
            super().__init__(_QTable.from_pandas(data), **kwargs)
        else:
            super().__init__(data, **kwargs)
        self._bckupSample = self.__create_backup_table()
        self.qinfo = None
        self.is_simulation = self.__check_simulation()
        self._merge_info: _gt.DataFrame = None
        self.zp_corrected: bool = False

    def __str__(self):
        """The string representation"""
        if self.is_simulation:
            return f"Simulated data sample" + "\n" + super().__str__()
        else:
            return str(self.gc) + "\n" + super().__str__()

    def __getattr__(self, attr: str):
        """The attribute getter"""
        # Allow attribute access to columns
        if attr in self.colnames:
            return self[attr]
        return super().__getattribute__(attr)

    def __contains__(self, item: str):
        """The item checker"""
        return item in self.colnames

    def __iter__(self):
        """The iterator"""
        return iter(self.colnames)

    def __reversed__(self):
        """The reversed iterator"""
        return reversed(self.colnames)
    
    def drop_columns(self, columns: list[str]):
        """
        Drops the specified columns from the sample data.

        Parameters
        ----------
        columns : list
            List of column names to drop.
        """
        self.remove_columns(columns)

    def head(self, n: int = 5):
        """Returns the first n rows of the sample"""
        return self.to_pandas().head(n)

    def describe(self):
        """Returns the description of the sample"""
        return self.to_pandas().describe()
    
    def to_pandas(self, *args, **kwargs) -> _gt.TabularData:
        """
        Converts the sample (`astropy.QTable` as default) to a pandas DataFrame.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the astropy.Table to_pandas method.
        **kwargs : dict
            Keyword arguments to pass to the astropy.Table to_pandas method.

        Returns
        -------
        df : pandas.DataFrame
            The DataFrame containing the sample data.
        """
        return super().to_pandas(*args, **kwargs)

    def to_numpy(self, columns: list[str] = None) -> _gt.Array:
        """
        Converts the sample data to a numpy array.

        Returns
        -------
        arr : numpy.ndarray
            The numpy array containing the sample data.
        """
        if columns is not None:
            return self.to_pandas()[columns].to_numpy()
        else:
            return self.to_pandas().to_numpy()

    def reset_sample(self) -> None:
        """
        Resets the sample to its original state.

        Returns
        -------
        None
        """
        # Remove all columns from self
        for col in list(self.colnames):
            self.remove_column(col)
        # Add columns from the backup QTable
        for col in self._bckupSample.itercols():
            self.add_column(col.copy())
    
    def _base_join(self, other: _gt.TabularData, keep: str = "both") -> _gt.TabularData:
        """
        Joins the sample data with another sample data.

        Parameters
        ----------
        other : TabularData
            The other sample data to join with.
        keep : str
            The type of join to perform. Can be 'left_only', 'right_only', or 'both'.
        inplace : bool
            If True, the operation is done in place, otherwise a new object is returned.

        Returns
        -------
        sample : TabularData
            The sample object containing the joined data.
        """
        sample = self.to_pandas()
        if not isinstance(other, _gt.DataFrame):
            other_sample = other.to_pandas()
        else:
            other_sample = other
        merged = sample.merge(other_sample, how="outer", indicator=True)
        if keep not in ["both", "left_only", "right_only"]:
            raise ValueError(
                "Invalid value for 'keep'. Must be 'both', 'left_only', or 'right_only'."
            )
        if keep != "both":
            merged = merged[merged["_merge"] == keep]
        merged_qtable = _QTable.from_pandas(merged)
        col_to_carry = merged_qtable.colnames[0]
        return merged_qtable, col_to_carry, merged


    def _dropna_inplace(self) -> None:
        """
        Drops rows with NaN values in the sample data.
        This method checks each column for NaN values and removes the rows
        with the maximum number of NaN values across all columns.
        """
        n_bad = {}
        for col in self.colnames:
            n_bad[col] = self.mask[col].sum()
        max_key = max(n_bad, key=n_bad.get)
        self.remove_rows(self.mask[max_key])
        return
    
    def _base_apply_conditions(self, conditions: str | list[str] | dict[str, str]) -> _gt.TabularData:
        """
        Applies conditions to the sample data.

        Parameters
        ----------
        conditions : str or list[str] or dict[str, str]
            The conditions to apply.
        inplace : bool
            If True, the operation is done in place, otherwise a new object is returned.

        Returns
        -------
        sample : grasp.Sample
            The sample object containing the filtered data.

        How to Use
        ----------
        The correct use of the method resides completely on how the conditions are passed.
        If passed as dictionary, the structure is as follows:
        it must have the column name as the key and the condition including the logical operation:
        >>> conditions = {
        ...     "parallax": "> -5",
        ...     "parallax": "< 5"
        ... }
        >>> sample.apply_conditions(conditions)

        If passed as a string, only one condition is admitted, in the form:
        >>> conditions = "parallax > -5"

        They can be passed as a list of strings, in the format:
        >>> conditions = ["parallax > -5", "parallax < 5"]
        """
        sample = (self.copy()).to_pandas()
        if isinstance(conditions, dict):
            conds = []
            for k, v in conditions.items():
                conds.append(f"(sample['{k}'] {v})")
            conds = " & ".join(conds)
        elif isinstance(conditions, list):
            conds = []
            for condition in conditions:
                conds.append(f"(sample.{condition})")
            conds = " & ".join(conds)
        elif isinstance(conditions, str):
            conds = f"(sample.{conditions})"
        else:
            raise ValueError("Conditions must be a dictionary, list or string.")
        mask = eval(conds)
        filtered_sample: _gt.DataFrame = sample[mask]
        return filtered_sample
    
    def _apply_conds_inplace(self, filtered_sample: _gt.DataFrame) -> None:
        # Store units before removing columns
        N_old=len(self)
        col_units = {
            col: self[col].unit if hasattr(self[col], "unit") else None
            for col in self.colnames
        }
        # Remove all columns
        for col in list(self.colnames):
            self.remove_column(col)
        # Add columns back, restoring units if present
        for col in filtered_sample.columns:
            unit = col_units.get(col, None)
            if unit is not None:
                self[col] = filtered_sample[col].values * unit
            else:
                self[col] = filtered_sample[col].values
        N_new = len(self)
        print(f"Cut {(1-N_new/N_old)*100:.3f}% of the sample")
        return

    def __check_simulation(self) -> bool:
        """Check wether the data the sample has been instanced with is
        a simulation or real data"""
        sim_a = [
            "Mass_[Msun]",
            "x_[pc]",
            "y_[pc]",
            "z_[pc]",
            "vx_[km/s]",
            "vy_[km/s]",
            "vz_[km/s]",
        ]
        if all([a == b for a, b in zip(sim_a, self.colnames)]):
            self.qinfo = "McLuster Simulation"
            self["M"] = self["Mass_[Msun]"] * _u.Msun
            self["x"] = self["x_[pc]"] * _u.pc
            self["y"] = self["y_[pc]"] * _u.pc
            self["z"] = self["z_[pc]"] * _u.pc
            self["vx"] = self["vx_[km/s]"] * _u.km / _u.s
            self["vy"] = self["vy_[km/s]"] * _u.km / _u.s
            self["vz"] = self["vz_[km/s]"] * _u.km / _u.s
            self.drop_columns(sim_a)
            is_simulation = True
        else:
            is_simulation = False
        return is_simulation


    def __create_backup_table(self) -> _gt.AstroTable:
        """
        Creates a backup of the sample data.

        Returns
        -------
        backup : astropy.Table
            The backup table containing the sample data.
        """
        new_table = _QTable()
        for col in self.colnames:
            new_table[col] = self[col]
        return new_table