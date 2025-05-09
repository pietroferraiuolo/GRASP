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

class SampleQTable(_QTable):
    """
    Subclass of QTable for handling the query result sample.
    Unifies the Cluster object and the sample data obtained.
    """
    def __init__(self, data: _gt.AstroTable = None, gc: _gt.Optional[_gt.GcInstance] = None, **kwargs: dict[str,_gt.Any]):
        # Accepts data as QTable, Table, DataFrame, or dict
        if isinstance(data, _pd.DataFrame):
            super().__init__(_QTable.from_pandas(data), **kwargs)
        else:
            super().__init__(data, **kwargs)
        self.qinfo = None
        self.is_simulation = self.__check_simulation()
        if isinstance(gc, str):
            if gc == 'UNTRACKEDDATA' or gc == 'UntrackedData':
                from astropy.units import deg
                self.gc = _Cluster('UntrackedData')
                self.gc.ra = self['ra'].mean()*deg
                self.gc.dec = self['dec'].mean()*deg
            else:
                self.gc = _Cluster(gc)
        else:
            self.gc = gc
        self._merge_info: _pd.DataFrame = None

    def __str__(self):
        if self.is_simulation:
            return f"Simulated data sample" + "\n" + super().__str__()
        else:
            return str(self.gc) + "\n" + super().__str__()

    def __repr__(self):
        return self.__get_repr()

    def __getattr__(self, attr: str):
        # Allow attribute access to columns
        if attr in self.colnames:
            return self[attr]
        return super().__getattribute__(attr)

    def __contains__(self, item: str):
        return item in self.colnames

    def __iter__(self):
        return iter(self.colnames)

    def __reversed__(self):
        return reversed(self.colnames)

    @property
    def sample(self):
        return self

    @property
    def meta(self):
        return super().meta
    metadata = meta

    def drop_columns(self, columns: list[str]):
        self.remove_columns(columns)

    def info(self, *args):
        return super().info(*args)

    def head(self, n: int = 5):
        return self.to_pandas().head(n)

    def describe(self):
        return self.to_pandas().describe()

    def join(self, other: "SampleQTable", keep: str = 'both', inplace: bool = False) -> "SampleQTable":
        sample = self.to_pandas()
        if not isinstance(other, _pd.DataFrame):
            other_sample = other.to_pandas()
        else:
            other_sample = other
        merged = sample.merge(other_sample, how="outer", indicator=True)
        if keep not in ['both', 'left_only', 'right_only']:
            raise ValueError("Invalid value for 'keep'. Must be 'both', 'left_only', or 'right_only'.")
        if keep != 'both':
            merged = merged[merged["_merge"] == keep]
        merged_qtable = _QTable.from_pandas(merged)
        if inplace:
            self.clear()
            for col in merged_qtable.colnames:
                self[col] = merged_qtable[col]
            self._merge_info = merged[["SOURCE_ID", "_merge"]]
            self.drop_columns(["_merge"])
            return self
        else:
            new_sample = SampleQTable(merged_qtable, self.gc)
            new_sample._merge_info = merged[["SOURCE_ID", "_merge"]]
            new_sample.drop_columns(["_merge"])
            return new_sample

    def to_pandas(self, *args, **kwargs) -> _gt.TabularData:
        return super().to_pandas(*args, **kwargs)

    def to_table(self, *args) -> _gt.TabularData:
        return _Table(self, *args)

    def to_numpy(self, columns: list[str] = None):
        if columns is not None:
            return self[columns].to_pandas().to_numpy()
        else:
            return self.to_pandas().to_numpy()

    def reset_sample(self):
        self.clear()
        for col in self._bckupSample.colnames:
            self[col] = self._bckupSample[col]

    def update_gc_params(self, **kwargs: dict[str,_gt.Any]) -> str:
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
        return str(self.gc)

    def apply_conditions(self, conditions: str|list[str]|dict[str,str], inplace: bool = False):
        sample = self.copy()
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
            self.clear()
            for col in sample.colnames:
                self[col] = sample[col]
            return self.__repr__()
        else:
            return SampleQTable(sample, self.gc)

    def __check_simulation(self) -> bool:
        sim_a = [
            'Mass_[Msun]',
            'x_[pc]',
            'y_[pc]',
            'z_[pc]',
            'vx_[km/s]',
            'vy_[km/s]',
            'vz_[km/s]'
        ]
        if all(a == b for a, b in zip(sim_a, self.colnames)):
            self.qinfo = "McLuster Simulation"
            self['M'] = self['Mass_[Msun]'] * _u.Msun
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
        from tabulate import tabulate
        if self.is_simulation:
            gctxt = f"""Simulated data sample"""
        elif self.gc and getattr(self.gc, 'id', None) == "UntrackedData":
            gctxt = f"""Gaia data retrieved at coordinates\nRA={self.gc.ra:.2f} DEC={self.gc.dec:.2f}\n"""
        else:
            gctxt = f"""Data sample for cluster {getattr(self.gc, 'id', 'Unknown')}\n"""
        stxt = "\n"
        names: list[str] = [name.lower() for name in self.colnames]
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
