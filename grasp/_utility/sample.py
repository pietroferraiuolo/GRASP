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
from .base_classes import BaseSample as _BaseSample
from grasp import types as _gt


class Sample(_BaseSample):
    """
    Subclass of QTable for handling the query result sample.
    Unifies the Cluster object and the sample data obtained.
    """

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
        super().__init__(data, **kwargs)


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
        merged_qtable,col_to_carry,merged = self._base_join(other, keep)
        if inplace:
            for col in list(self.colnames):
                self.remove_column(col)
            for col in merged_qtable.colnames:
                self[col] = merged_qtable[col]
            self._merge_info = merged[[col_to_carry, "_merge"]]
            self.drop_columns(["_merge"])
            return self
        else:
            new_sample = Sample(merged_qtable)
            new_sample.meta = self.meta.copy()
            new_sample._merge_info = merged[[col_to_carry, "_merge"]]
            new_sample.drop_columns(["_merge"])
            return new_sample
    
    def dropna(self, inplace: bool = True) -> None | _gt.TabularData:
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
            self._dropna_inplace()
            return
        else:
            df = (self.copy()).to_pandas()
            df.dropna(inplace=True)
            return Sample(_QTable.from_pandas(df))
    

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
        filtered_sample: _gt.DataFrame = self._base_apply_conditions(conditions)
        if inplace:
            self._apply_conds_inplace(filtered_sample)
            return
        else:
            return Sample(filtered_sample)


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


class GcSample(_BaseSample):
    """
    Subclass of Sample for handling the globular cluster sample.
    It is a Sample with the gc attribute set to a Cluster object.
    """

    def __init__(self, data: _gt.AstroTable = None, gc: _gt.GcInstance = None, **kwargs: dict[str, _gt.Any]):
        """
        Class for handling the globular cluster sample.

        Parameters
        ----------
        data : astropy.Table, pandas.DataFrame, dict
            Table containing the retrieved sample's data.
        gc : grasp.cluster.Cluster
            Globular cluster object used for the query.
        """
        super().__init__(data, **kwargs)
        if isinstance(gc, str):
            if gc == "UNTRACKEDDATA" or gc == "UntrackedData":
                from astropy.units import deg

                self.gc = _Cluster("UntrackedData")
                if "ra" in self.colnames and "dec" in self.colnames:
                    self.gc.ra = self["ra"].mean() * deg
                    self.gc.dec = self["dec"].mean() * deg
                else:
                    self.gc.ra = 0.0
                    self.gc.dec = 0.0
            else:
                self.gc = _Cluster(gc)
        else:
            self.gc = gc
    
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
        merged_qtable,col_to_carry,merged = self._base_join(other, keep)
        if inplace:
            for col in list(self.colnames):
                self.remove_column(col)
            for col in merged_qtable.colnames:
                self[col] = merged_qtable[col]
            self._merge_info = merged[[col_to_carry, "_merge"]]
            self.drop_columns(["_merge"])
            return self
        else:
            new_sample = GcSample(merged_qtable, self.gc)
            new_sample.meta = self.meta.copy()
            new_sample._merge_info = merged[[col_to_carry, "_merge"]]
            new_sample.drop_columns(["_merge"])
            return new_sample
            
    
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

    def dropna(self, inplace: bool = True) -> None | _gt.TabularData:
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
            self._dropna_inplace()
            return
        else:
            df = (self.copy()).to_pandas()
            df.dropna(inplace=True)
            return GcSample(_QTable.from_pandas(df), self.gc)

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
        filtered_sample: _gt.DataFrame = self._base_apply_conditions(conditions)
        if inplace:
            self._apply_conds_inplace(filtered_sample)
            return
        else:
            return GcSample(filtered_sample, self.gc)

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