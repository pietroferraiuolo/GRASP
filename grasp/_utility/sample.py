"""
::module..grasp._utility.sample_qtable

sample_qtable.py
================

Subclass of astropy QTable with custom Sample features.

Author(s):
----------
- Pietro Ferraiuolo : Written in 2024 (adapted from Sample)

Description:
------------
This module provides a subclass of QTable for handling the query result sample.
It unifies the Cluster object and the sample data obtained, in order to have a
compact object which has everything and from which quantities can be computed
easily, while behaving as a QTable.
"""

import pandas as _pd
from astropy import units as _u
from grasp._utility.cluster import Cluster as _Cluster
from astropy.table import QTable as _QTable, Table as _Table
from grasp import types as _gt


class Sample(_QTable):
    """
    Subclass of QTable for handling the query result sample.
    Unifies the Cluster object and the sample data obtained.
    """

    def __init__(
        self,
        data: _gt.AstroTable = None,
        gc: _gt.Optional[_gt.GcInstance] = None,
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
        if isinstance(data, _pd.DataFrame):
            super().__init__(_QTable.from_pandas(data), **kwargs)
        else:
            super().__init__(data, **kwargs)
        self._bckupSample = self.__create_backup_table()
        self.qinfo = None
        self.is_simulation = self.__check_simulation()
        if isinstance(gc, str):
            if gc == "UNTRACKEDDATA" or gc == "UntrackedData":
                from astropy.units import deg

                self.gc = _Cluster("UntrackedData")
                if "ra" in self.colnames and "dec" in self.colnames:
                    self.gc.ra = self["ra"].mean() * deg
                    self.gc.dec = self["dec"].mean() * deg
                else:
                    self.gc.ra = 0.
                    self.gc.dec = 0.
            else:
                self.gc = _Cluster(gc)
        else:
            self.gc = gc
        self._merge_info: _pd.DataFrame = None

    def __str__(self):
        """The string representation"""
        if self.is_simulation:
            return f"Simulated data sample" + "\n" + super().__str__()
        else:
            return str(self.gc) + "\n" + super().__str__()

    def __repr__(self):
        """The representation"""
        return self.__get_repr()

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


    def join(
        self, other: _gt.TabularData, keep: str = "both", inplace: bool = False
    ) -> _gt.TabularData:
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
        if not isinstance(other, _pd.DataFrame):
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
        if inplace:
            for col in list(self.colnames):
                self.remove_column(col)
            for col in merged_qtable.colnames:
                self[col] = merged_qtable[col]
            self._merge_info = merged[[col_to_carry, "_merge"]]
            self.drop_columns(["_merge"])
            return self
        else:
            new_sample = Sample(merged_qtable, self.gc)
            new_sample._merge_info = merged[[col_to_carry, "_merge"]]
            new_sample.drop_columns(["_merge"])
            return new_sample

    def dropna(self, inplace: bool = True) -> None|_gt.TabularData:
        """
        Drops rows with NaN values from the sample data.

        Parameters
        ----------
        inplace : bool
            If True, the operation is done in place, otherwise a new object is returned.
        
        Returns
        -------
        sample : grasp.Sample or None
            The sample object containing the filtered data, or None if inplace is True.
        """
        if inplace:
            n_bad = {}
            for col in self.colnames:
                n_bad[col] = self.mask[col].sum()
            max_key = max(n_bad, key=n_bad.get)
            self.remove_rows(self.mask[max_key])
            return
        else:
            df = (self.copy()).to_pandas()
            df.dropna(inplace=True)
            return Sample(_QTable.from_pandas(df), self.gc)


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
            return self[columns].to_pandas().to_numpy()
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


    def update_gc_params(self, **kwargs: dict[str, _gt.Any]) -> None:
        """
        Updates the parameters of the cluster object.

        Parameters
        ----------
        **kwargs : dict
            The parameters to update.
        """
        if self.is_simulation:
            return "This is a simulation data sample. No GC available."
        for key in kwargs:
            if hasattr(self.gc, key):
                setattr(self.gc, key, kwargs[key])
            else:
                if not self.gc.id == "UntrackedData":
                    text = self.__get_repr()
                    text = text.split("\n")[5:]
                    ptxt = "\n".join(text)
                else:
                    ptxt = ""
                raise AttributeError(
                    f"'Cluster' object has no attribute '{key}'\n{ptxt}"
                )
        print(self.gc.__str__())
        return


    def apply_conditions(
        self, conditions: str | list[str] | dict[str, str], inplace: bool = False
    ) -> _gt.TabularData:
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
        if inplace:
            #meta_bck = self.meta.copy()
            # Remove all columns and add filtered columns
            for col in list(self.colnames):
                self.remove_column(col)
            for col in filtered_sample.columns:
                self[col] = filtered_sample[col]
            #self = QTable.from_pandas(filtered_sample)
            #self.meta = meta_bck
            print(self.__repr__())
            return
        else:
            return Sample(filtered_sample, self.gc)


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


    def __get_repr(self) -> str:
        """Gets the str representation"""
        from tabulate import tabulate

        if self.is_simulation:
            gctxt = f"""Simulated data sample"""
        elif self.gc.id == "UntrackedData":
            gctxt = f"""Gaia data retrieved at coordinates
RA={self.gc.ra:.2f} DEC={self.gc.dec:.2f}
"""
        else:
            gctxt = f"""Data sample for cluster {self.gc.id}
"""
        stxt = "\n"  # "\nData Columns:\n"
        names: list[str] = [name.lower() for name in self.colnames]
        max_len: int = len(max(names, key=len))
        terminal_min_size: int = 80
        ncols: int
        if terminal_min_size / max_len < 1.5:
            ncols = 2
        else:
            from math import ceil

            ncols = ceil(terminal_min_size / max_len)
        righe: list[str] = [names[i : i + ncols] for i in range(0, len(names), ncols)]
        headers = ["Data Columns:"] + [""] * (ncols - 1)
        tabula: str = tabulate(righe, headers=headers, tablefmt="presto")
        stxt += tabula.replace("|", " ").replace("+", "-")
        return gctxt + stxt


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