#from __future__ import annotations
"""
Author(s)
---------
    - Pietro Ferraiuolo : Written in 2024

Description
-----------
A module which contains the Cluster class, which contains all the information of
a specified cluster.

How to Use
----------
Initialize the class with a cluster's name. As example

>>> from grasp.cluster import Cluster
>>> ngc104 = Cluster('ngc104')

Now we can call methods to read the parameters

>>> ngc104.id
'NGC104'
>>> ngc104.w0
8.82
"""

import os, shutil, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from grasp.core.folder_paths import (
    CLUSTER_DATA_FOLDER,
    CLUSTER_MODEL_FOLDER,
    CATALOG_FILE,
    UNTRACKED_DATA_FOLDER,
)
from grasp.plots import label_font, title_font


class Cluster:
    """
    Class for the cluster parameter loading.

    Upon initializing, it is loaded with the specified cluster's parameters, ta
    ken from the Harris Catalogue 2010 Edition.

    Methods
    -------
    _loadClusterParameters(self, name) : function
        Loads the desired cluster's parameters

    How to Use
    ----------
    Initialize the class with a cluster's name. As example

    >>> from grasp._cluster import Cluster
    >>> ngc104 = Cluster('ngc104')
    """

    def __init__(self, name: str = None, **params):
        """The constructor"""
        if name is not None:
            self.id = name.upper()
            self.data_path = CLUSTER_DATA_FOLDER(self.id)
            self.model_path = CLUSTER_MODEL_FOLDER(self.id)
            parms = self._load_cluster_parameters(self.id)
            self.ra = parms.loc["ra"] * u.deg
            self.dec = parms.loc["dec"] * u.deg
            self.dist = parms.loc["dist"] * u.kpc
            self.rc = parms.loc["rc"] / 60 * u.deg
            self.rh = parms.loc["rh"] / 60 * u.deg
            self.w0 = parms.loc["w0"]
            self.logc = parms.loc["logc"]
            self.rt = self.rc * 10**self.logc
            self.cflag = ["True " if parms.loc["collapsed"] == "Y" else False][0]
            self.model = self._load_king_model()
        else:
            print("Not a Cluster: no model available")
            self.data_path = UNTRACKED_DATA_FOLDER
            self.id = "UntrackedData"
            self.ra = params.get("ra", None)
            self.dec = params.get("dec", None)
            self.model = None

    def __str__(self):
        """String representation"""
        return self.__get_str()

    def __repr__(self):
        """Representation"""
        return self.__get_repr()

    def show_model(self, **kwargs):
        """
        Function for plotting the loaded king model.

        Optional Parameters
        -------------------
        **kwargs :
            color : color of the main plot.
            scale : scale of the axes, default linear.
            grid  : grid on the plot
        """
        scale = kwargs.get("scale", None)
        xscale = kwargs.get("xscale", None)
        yscale = kwargs.get("yscale", None)
        c = kwargs.get("color", "black")
        grid = kwargs.get("grid", False)
        plt.figure(figsize=(8, 6))
        plt.plot(self.model["xi"], self.model["w"], color=c)
        plt.plot(
            [self.model["xi"].min(), self.model["xi"].min()],
            [self.model["w"].min() - 1, self.model["w"].max()],
            c="red",
            linestyle="--",
            label=rf"$W_0$={self.model['w'].max():.2f}",
        )
        plt.xlabel(r"$\xi$ = $\dfrac{r}{r_t}$", fontdict=label_font)
        plt.ylabel("w", fontdict=label_font)
        plt.title("Integrated King Model", fontdict=title_font)
        plt.ylim(-0.2, self.model["w"].max() + 0.2)
        plt.xlim(-0.05, 1.05)
        if grid:
            plt.grid()
        if xscale is not None or scale is not None:
            plt.xscale(scale)
        if yscale is not None or scale is not None:
            plt.yscale(scale)
        plt.legend(loc="best")
        plt.show()

    def _load_cluster_parameters(self, name: str):
        """
        Loads the parameters of the requested cluster from the Harris Catalog
        2010 Globular Cluster Database, written in the Catalogue.xlsx file

        Parameters
        ----------
        name : str
            Name of the requested Globular Cluster.

        Returns
        -------
        cat_row : TYPE
            Pandas Series with all the necessary paramenters to ilitialize the Cluster Class.
        """
        catalog = pd.read_excel(CATALOG_FILE, index_col=0)
        cat_row = catalog.loc[name.upper()]
        return cat_row

    def _load_king_model(self):
        """
        Loads the integrated Single-Mass King model for the cluster.

        Returns
        -------
        model : astropy table
            Astropy table containing the integrated quantities of the king model.
            These are:
                'xi': dimentionless radial distance from gc center, normalized
                at tidal radius.
                'w': the w-king parameter, that is essentially the gravitational
                potential.
                'rho': the noralized density profile of the clusted.
        """
        model = Table()
        try:
            try:
                file = os.path.join(CLUSTER_MODEL_FOLDER(self.id), "SM_king.txt")
                model["xi"] = np.loadtxt(file, skiprows=1, usecols=1)
                model["w"] = np.loadtxt(file, skiprows=1, usecols=2)
                model["rho"] = np.loadtxt(file, skiprows=1, usecols=3)
            except Exception:
                import io, re
                file = os.path.join(CLUSTER_MODEL_FOLDER(self.id), "SM_king.txt")
                with open(file, "r") as f:
                    content = f.read()
                content_fixed = re.sub(r'(?<=[0-9])-(?=\d)', r' -', content)
                model['xi'] = np.loadtxt(io.StringIO(content_fixed), skiprows=1, usecols=1)
                model['w'] = np.loadtxt(io.StringIO(content_fixed), skiprows=1, usecols=2)
                model['rho'] = np.loadtxt(io.StringIO(content_fixed), skiprows=1, usecols=3)
        except FileNotFoundError:
            from grasp.analyzers.king import king_integrator
            print(
                f"WARNING: no king model file found for '{self.id}'. Performing the Single-Mass King model integration."
            )
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            result = king_integrator(self.w0)
            mod = pd.read_csv(result, delim_whitespace=True, skipfooter=1)
            model["xi"] = mod.xi
            model["w"] = mod.w
            model["rho"] = mod.rho_rho0
            shutil.move(result, os.path.join(self.model_path, "SM_king.txt"))
        return model

    def __get_repr(self):
        """repr creation"""
        if self.id == "UntrackedData":
            text = "<grasp._cluster.Cluster object>"
        else:
            text = f"<grasp._cluster.Cluster object: {self.id}>"
        return text

    def __get_str(self):
        """str creation"""
        if self.id == "UntrackedData":
            text = f"""
Scansion at RA {self.ra:.3f} DEC {self.dec:.3f}
"""
        else:
            text = f"""
Harris Catalog 2010 edition Parameters

       Key                  Value
----------------------------------------
.id      Cluster Name       {self.id}
.ra      Position in sky    RA  {self.ra:.2f}
.dec                        DEC {self.dec:.2f}
.dist    Distance           {self.dist:.2f}
.w0      W0 Parameter       {self.w0}
.logc    Concentration      logc={self.logc:.2f}
.cflag                      Collapsed -> {self.cflag}
.rc      Core radius        {self.rc:.3f}
.rh      Half-Light radius  {self.rh:.3f}
.rt      Tidal Radius       {self.rt:.3f}
"""
        return text
