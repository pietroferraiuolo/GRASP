"""
::module..grasp._utility.sample

sample.py
=========


Author(s):
----------
- Pietro Ferraiuolo : Written in 2024

Description:
------------
This module provides a class for handling the query result sample. The class
unifies the Cluster object and the sample data obtained, in order to have a
compact object which has everything and from which quantities can be computed
easily.


"""

import pandas as _pd
from astropy import units as _u
from grasp._utility.cluster import Cluster as _Cluster
from astropy.table import QTable as _QTable, Table as _Table
from grasp import types as _gt


class Sample:
    """
    Class for handling the query result sample.

    It is an object with unifies the Cluster object and the sample data obtained,
    in order to have a compact object which has everythin and from which quantities
    can be computed easily.

    Parameters
    ----------
    sample : astropy.table.Table
        Table containing the retrieved sample's data.
    gc : grasp.cluster.Cluster, optional
        Globular cluster object used for the query.
    """

    def __init__(self, sample: _gt.AstroTable, gc: _gt.Optional[_gt.GcInstance] = None):
        """The constructor"""
        self.qinfo = None
        self._sample = (
            _QTable.from_pandas(sample) if isinstance(sample, _pd.DataFrame) else sample
        )
        self._table = None
        self.is_simulation = self.__check_simulation()
        self._bckupSample = self._sample.copy()
        if isinstance(gc, _Cluster):
            self.gc = gc
        elif isinstance(gc, str):
            if gc == 'UNTRACKEDDATA' or gc == 'UntrackedData':
                from astropy.units import deg
                self.gc = _Cluster('UntrackedData')
                self.gc.ra = self._sample['ra'].mean()*deg
                self.gc.dec = self._sample['dec'].mean()*deg
            else:
                self.gc = _Cluster(gc)
        else:
            self.gc = None
        self._merge_info: _pd.DataFrame = None

    def __str__(self):
        """The string representation"""
        if self._is_simulation:
            return f"Simulated data sample"+ "\n" + self._sample.__str__()
        else:
            return self.gc.__str__() + "\n" + self._sample.__str__()

    def __repr__(self):
        """The representation"""
        return self.__get_repr()

    def __len__(self):
        """The length of the sample"""
        return len(self._sample["SOURCE_ID"])

    def __getitem__(self, key: str):
        """The item getter"""
        return self._sample[key]

    def __getattr__(self, attr: str):
        """The attribute getter"""
        try:
            if attr in self.__colnames__():
                return self._sample[attr]
            else:
                getattr(self._sample, attr)
        except Exception as e:
            raise(e)

    def __setitem__(self, key: str, value: _gt.Any):
        """The item setter"""
        self._sample[key] = value

    def __contains__(self, item: str):
        """The item checker"""
        return item in self.__colnames__()

    def __iter__(self):
        """The iterator"""
        return iter(self.__colnames__())

    def __reversed__(self):
        """The reversed iterator"""
        return reversed(self.__colnames__())
    
    def __colnames__(self):
        """The column names"""
        if isinstance(self._sample, _pd.DataFrame):
            return self._sample.columns.values
        elif isinstance(self._sample, (_Table,_QTable)):
            return self._sample.colnames

    @property
    def sample(self):
        """Returns the sample data"""
        return self._sample
    
    @property
    def meta(self):
        """Returns the metadata of the sample"""
        return self._sample.meta
    metadata = meta

    def drop_columns(self, columns: list[str]):
        """
        Drops the specified columns from the sample data.

        Parameters
        ----------
        columns : list
            List of column names to drop.
        """
        self._sample.remove_columns(columns)


    def info(self, *args):
        """Returns the info of the sample"""
        return self._sample.info(*args)
    

    def head(self, n: int = 5):
        """Returns the first n rows of the sample"""
        return self._sample.to_pandas().head(n)
    

    def describe(self):
        """Returns the description of the sample"""
        return self._sample.to_pandas().describe()


    def join(self, other: "Sample", keep: str = 'both', inplace: bool = False) -> _gt.TabularData:
        """
        Joins the sample data with another sample data.

        Parameters
        ----------
        other : grasp.Sample
            The other sample data to join with.
        keep : str
            The type of join to perform. Can be 'left_only', 'right_only', or 'both'.
        inplace : bool
            If True, the operation is done in place, otherwise a new object is returned.

        Returns
        -------
        sample : grasp.Sample
            The sample object containing the joined data.
        """
        sample = self._sample.to_pandas()
        other_sample = other.to_pandas()
        merged = sample.merge(other_sample, how="outer", indicator=True)
        if keep not in ['both', 'left_only', 'right_only']:
            raise ValueError("Invalid value for 'keep'. Must be 'both', 'left_only', or 'right_only'.")
        if keep != 'both':
            merged = merged[merged["_merge"] == keep]
        merged_qtable = _QTable.from_pandas(merged)
        if inplace:
            self._sample = merged_qtable
            self._merge_info = merged[["SOURCE_ID", "_merge"]]
            self.drop_columns(["_merge"])
            return merged_qtable
        else:
            new_sample = Sample(merged_qtable, self.gc)
            new_sample._merge_info = merged[["SOURCE_ID", "_merge"]]
            new_sample.drop_columns(["_merge"])
            return new_sample


    def to_pandas(self, overwrite: bool = False, *args: tuple[str,_gt.Any], **kwargs: dict[str,_gt.Any]) -> _gt.TabularData:
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
        if isinstance(self._sample, (_QTable, _Table)):
            pandas_df = self._sample.to_pandas(*args, **kwargs)
            if overwrite:
                self._table = self._sample.copy()
                self._sample = pandas_df
                print(self._sample.__repr__)
                return self
            return pandas_df
        else:
            raise TypeError(
                f"Expected an astropy.Table or QTable, but got {type(self._sample)}"
            )


    def to_table(self, *args: tuple[str,_gt.Any]) -> _gt.TabularData:
        """
        Converts back the sample from a pandas.DataFrame into an astropy Table.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the astropy.Table constructor.

        Returns
        -------
        table : astropy.Table
            The table containing the sample data.
        """
        if isinstance(self._sample, _pd.DataFrame):
            self._sample = _Table.from_pandas(self._sample, *args)
            return self._sample.__repr__()
        else:
            raise TypeError(
                f"Expected a pandas.DataFrame, but got {type(self._sample)}"
            )


    def to_numpy(self, columns: list[str] = None):
        """
        Converts the sample data to a numpy array.

        Returns
        -------
        arr : numpy.ndarray
            The numpy array containing the sample data.
        """
        if columns is not None:
            return self._sample[columns].to_pandas().to_numpy()
        else:
            return self._sample.to_pandas().to_numpy()


    def reset_sample(self):
        """Resets the sample to its original state"""
        self._sample = self._bckupSample.copy()


    def update_gc_params(self, **kwargs: dict[str,_gt.Any]) -> str:
        """
        Updates the parameters of the cluster object.

        Parameters
        ----------
        **kwargs : dict
            The parameters to update.
        """
        if self._is_simulation:
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
        return self.gc.__str__()
    

    def apply_conditions(self, conditions: str|list[str]|dict[str,str], inplace: bool = False):
        """
        Applies conditions to the sample data.

        Parameters
        ----------
        conditions : dict
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
        sample = self._sample.copy()
        if isinstance(conditions, dict):
            conds = []
            for k,v in conditions.items():
                conds.append(f"(sample.{k} {v})")
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
        sample = sample[eval(conds)]
        if inplace:
            self._sample = sample
            return self.__repr__()
        else:
            return Sample(sample, self.gc)


    def __check_simulation(self) -> bool:
        """Check wether the data the sample has been instanced with is
        a simulation or real data"""
        sim_a = [
            'Mass_[Msun]',
            'x_[pc]',
            'y_[pc]',
            'z_[pc]',
            'vx_[km/s]',
            'vy_[km/s]',
            'vz_[km/s]'
        ]
        if all(a == b for a, b in zip(sim_a, self.__iter__())):
            self.qinfo = "McLuster Simulation"
            self._sample['M'] = self._sample['Mass_[Msun]'] * _u.Msun
            self._sample["x"] = self._sample["x_[pc]"] * _u.pc
            self._sample["y"] = self._sample["y_[pc]"] * _u.pc
            self._sample["z"] = self._sample["z_[pc]"] * _u.pc
            self._sample["vx"] = self._sample["vx_[km/s]"] * _u.km / _u.s
            self._sample["vy"] = self._sample["vy_[km/s]"] * _u.km / _u.s
            self._sample["vz"] = self._sample["vz_[km/s]"] * _u.km / _u.s
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
        stxt = "\n" # "\nData Columns:\n"
        names: list[str] = [name.lower() for name in self.__colnames__()]
        max_len: int = len(max(names, key=len))
        terminal_min_size: int = 80
        ncols: int
        if terminal_min_size/max_len <1.5:
            ncols = 2
        else:
            from math import ceil
            ncols = ceil(terminal_min_size/max_len)
        righe: list[str] = [names[i:i+ncols] for i in range(0, len(names), ncols)]
        headers = ['Data Columns:']+['']*(ncols-1)
        tabula: str = tabulate(righe, headers=headers, tablefmt='presto')
        stxt += tabula.replace('|', ' ').replace('+','-')
        return gctxt + stxt
