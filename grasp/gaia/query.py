"""
Module: query.py

Author(s)
---------
- Pietro Ferraiuolo : Written in 2024

Description
-----------
Thi module contains the GaiaQuery class, which handles the ADQL language to make
queries for globular clusters data retrievement fast and easy.

How to Use it
-------------
After importing the module, initialize the class with a table (default is the
Gaia data release 3 table)

    >>> from grasp.gaia import query
    >>> gq = query.GaiaQuery()
    >>> Initialized with Gaia table: 'gaiadr3.gaia_source'
    

To check all the available Gaia mission tables

    >>> query.available_tables()
    INFO: Retrieving tables... [astroquery.utils.tap.core]
    INFO: Parsing tables... [astroquery.utils.tap.core]
    INFO: Done. [astroquery.utils.tap.core]
    external.apassdr9
    external.catwise2020
    ...

To show information about the loaded table(s), one can get the table description
with the '__repr__' method, that is:

    >>> gq
    GAIADR3.GAIA_SOURCE
    -------------------
    This table has an entry for every Gaia observed source as published with this
    data release. It contains the basic source parameters, in their final state
    as processed by the Gaia Data Processing and Analysis Consortium from the raw
    data coming from the spacecraft. The table is complemented with others containing
    information specific to certain kinds of objects
    (e.g.~Solar--system objects, non--single stars, variables etc.) and value--added
    processing (e.g.~astrophysical parameters etc.). Further array data types
    (spectra, epoch measurements) are presented separately via Datalink resources.

While for a complete list of parameters within the table, print out the query
object:

    >>> print(gq)
    0 solution_id
    1 designation
    2 source_id
    3 random_index
    4 ref_epoch
    .
    .
    .
    150 ebpminrp_gspphot_upper
    151 libname_gspphot
"""

import os as _os
import numpy as _np
import configparser as _cp
from astropy import units as _u
from astroquery.gaia import Gaia as _Gaia
from grasp._utility.sample import Sample as _Sample
from typing import Optional as _Opt, Union as _Union
from grasp._utility.cluster import Cluster as _Cluster
from astropy.table import Table as _Table, QTable as _QTable
from grasp.core.osutils import (
    get_kwargs,
    timestamp as _ts,
    load_data as _loadData,
    tnlist as _tnlist
)
from grasp.core.folder_paths import (
    BASE_DATA_PATH as _BDP,
    CLUSTER_DATA_FOLDER as _CDF,
    UNTRACKED_DATA_FOLDER as _UD,
    )

_QDATA = "query_data.fits"
_QINFO = "query_info.ini"


def available_tables(key: str = None):
    """
    Prints out the complete list of data tables present in the Gaia archive.

    Parameters
    ----------
    key : str, optional
        A key used to restrict the printed tables. As example, if
        >>> key = 'gaiadr3'
        then only tables relative to the complete 3th data release will be printed
        out. Default is None, meaning all the tables will be printed.
    """
    tables = _Gaia.load_tables(only_names=True)
    if key is not None:
        for table in tables:
            name = table.name
            if key in name:
                print(name)
    else:
        for table in tables:
            print(table.name)


class GaiaQuery:
    """
    Classs for the Gaia Query language execution.

    Description
    -----------
    With this class, it is possible to easily perform async queries and retriev
    e data from the ESA/GAIA catalogue. It is possible to use different data re
    leases by loading different data tables in the initialization of the class.

    Methods
    -------
    free_query:
        Perform an ADQL search into the Gaia catalogue with custom data to retrieve
        and conditions to apply.
    get_atrometry:
        A pre-constructed ADQL search into the Gaia catalogue with fixed data
        retrieved, which is the principal astrometric parameters, with the possibility
        to add personalized query conditions.
    get_photometry:
        A pre-constructed ADQL search into the Gaia catalogue with fixed data
        retrieved, which is the principal photometric parameters, with the possibility
        to add personalized query conditions.
    get_rv:
        A pre-constructed ADQL search into the Gaia catalogue with fixed data
        retrieved, which is the radial velociti parameter with its error, with
        the possibility to add personalized query conditions.

    How to Use it
    -------------
    Import the class and initialize it

    >>> from grasp.query import GaiaQuery
    >>> dr3 = GaiaQuery()
    'Initialized with Gaia table: gaiadr3.gaia_source'

    To use a different Gaia catalogue simply initialize the class to it:

    >>> table = 'gaiadr2.gaia_source'
    >>> dr2 = GaiaQuery(gaia_table=table)
    'Initialized with Gaia table: gaiadr2.gaia_source'

    The queries, to work efficiently, require a 'grasp.cluster.Cluster' object: so,
    let's take an example cluster, ngc104:

        >>> from grasp._cluster import Cluster
        >>> gc = Cluster('ngc104')
        >>> gc
        <grasp.cluster.Cluster object: NGC104>

    At this point, simply passing as argument the gc object to the query function:

        >>> data = dr3.get_astrometry(gc, radius=0.1) # Radius must be passed as deg
        INFO: Query finished. [astroquery.utils.tap.core]
        Sample number of sources: 45865
        >>> data
             SOURCE_ID              ra         ...        pmdec        pmdec_error
                                   deg         ...       mas / yr        mas / yr
               int64             float64       ...       float64         float32
        ------------------- ------------------ ... ------------------- -----------
        4689637445779054208   5.76563951251253 ...  -2.259232804525973  0.27563116
        4689638850232458368  5.874682871570303 ... -2.3177094407812033  0.17122078
                ...                ...         ...         ...             ...

    The methods also have the save option, to save both the result of the query
    and its information, such as the parameters used.

    """

    def __init__(self, gaia_table: _Opt[_Union[str, list]] = "gaiadr3.gaia_source"):
        """
        The Constructor

        Parameters
        ----------
        gaia_table : str or list of str, optional
            Gaia table(s) to initialize the class with. The default is the 3th
            Gaia data release "gaiadr3.gaia_source".
        """
        _Gaia.MAIN_GAIA_TABLE = gaia_table
        _Gaia.ROW_LIMIT = -1
        self._table = gaia_table
        self._path = _BDP
        self._fold = None
        self._queryInfo = {}
        self._baseQ = """SELECT {data}
FROM {table}
WHERE CONTAINS(POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),CIRCLE('ICRS',{circle}))=1
    {cond}"""
        self.last_result = None
        self.last_query = None
        print(f"Initialized with Gaia table: '{gaia_table}'")


    def __repr__(self):
        """The representation"""
        return self.__get_repr()


    def __str__(self):
        """The string representation"""
        return self.__get_str()
    

    @property
    def parameters(self):
        """The parameters"""
        print(self.__params())
    

    @property
    def description(self):
        """The description"""
        print(self.__descr())
    

    def free_query(self, adql_cmd: str, save: bool = False):
        """
        Function to perform a query with the ADQL language, using the
        GaiaQuery class.

        Parameters
        ----------
        adql_cmd : str
            The ADQL command to execute.
            _Hint_: It is higly recommended to use the triple quote \""" to
            write the command, in order to avoid problems with the
            indentation and the line breaks.

            Example:
            ```python
            adql_cmd = '''SELECT source_id, ra, dec
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',ra,dec,radius))=1'''
            ```

            Refer to the 
            <a href=https://www.cosmos.esa.int/web/gaia-users/archive/writing-queries#query_example_1>
            ESA Cosmos</a> documentation for query examples, and to the 
            <a href=https://ivoa.net/documents/ADQL/20230418/PR-ADQL-2.1-20230418.html#tth_sEc4.2.9>
            ADQL</a> documentation of syntax and functions.

        Returns
        -------
        result : astropy.table.QTable
            The result of the query.
        """
        job = _Gaia.launch_job_async(adql_cmd)
        result = job.get_results()
        name = 'UntrackedData'
        self._queryInfo = {
            "Scan Info": {'Command': adql_cmd},
            "Flag": {"Query": "'true free'"},
        }  
        if save:
            self._saveQuery(result, name)
        return result

    def free_gc_query(
        self,
        radius,
        gc: _Opt[_Union[_Cluster, str]] = None,
        save: bool = False,
        **kwargs,
    ):
        """
        This function allows to perform an ADQL search into the Gaia catalogue with
        personalized parameters, such as data to collect and conditions to apply.

        Parameters
        ----------
        radius : float
            Radius, in degrees, of the scan circle.
        gc : grasp.cluster.Cluster or str
            String name or Cluster object, created with the G-GCAS module, of a globular cluster.
        save : bool, optional
            Whether to save the obtained data with its information or not.
        **kwargs : additional optional arguments
        ra : float or str
            Right ascension coordinate for the centre of the scan (if no gc is provided).
        dec : float or str
            Declination coordinate for the centre of the scan (if no gc is provided)
        name : str
            String which provides the folder name where to save the data.
            Needed if no 'gc' object is supplied: if it is not given, and
            save was True, the data will be stored in the 'UntrackedData'
            folder.
        data : str or list of str
            List of parameters to retrieve, from the ones printed by ''.print_table()''.
            If this argument is missing, the only parameter retrieved is 'source_id'.
            Aliases:
            - 'dat'
            - 'data'
            - 'params'
            - 'parameters'
        conditions : str or list of str
            Listo of conditions on the parameters to apply upon scanning the
            archive. If no conditions are supplied, no conditions are applied.
            Aliases:
            - 'cond'
            - 'conds'
            - 'condition'
            - 'conditions'

        Returns
        -------
        job_result : astropy.table.QTable
            Result of the async query, stored into an astropy table.

        """
        ra = kwargs.get("ra", None)
        dec = kwargs.get("dec", None)
        name = kwargs.get("name", "UntrackedData")
        gc, ra, dec, savename = self._get_coordinates(gc, ra=ra ,dec=dec ,name=name)
        self._queryInfo = {
            "Scan Info": {"RA": ra, "DEC": dec, "Scan Radius": radius},
            "Flag": {"Query": "free"},
        }
        dat = get_kwargs(("data", "dat", "params", "parameters"), "source_id", kwargs)
        self._queryInfo["Scan Info"]["Data Acquired"], _ = self._formatCheck(
            dat, "None"
        )
        cond = get_kwargs(("cond", "conds", "conditions", "condition"), "None", kwargs)
        if isinstance(cond, list):
            ccond = ""
            for c in range(len(cond) - 1):
                ccond += cond[c] + ", "
            ccond += cond[-1]
            self._queryInfo["Scan Info"]["Conditions Applied"] = ccond
        else:
            self._queryInfo["Scan Info"]["Conditions Applied"] = cond
        samp = self._run_query(savename, ra, dec, radius, dat, cond, save)
        sample = _Sample(samp, gc=gc)
        sample.qinfo = self._queryInfo["Scan Info"]
        return sample

    def get_astrometry(
        self,
        radius,
        gc: _Opt[_Union[_Cluster, str]] = None,
        save: bool = False,
        **kwargs,
    ):
        """
        A pre-constructed ADQL search into the Gaia catalogue with fixed data
        retrieved, which is the principal astrometric parameters, with the possibility
        to add personalized query conditions.

        The retrieved data is:
        'source_id, ra, ra_error, dec, dec_error, parallax, parallax_error, pmra,
        pmra_error, pmdec, pmdec_error'

        Parameters
        ----------
        radius : float
            Radius, in degrees, of the scan circle.
        gc : grasp.cluster.Cluster or str
            String name or Cluster object created with the G-GCAS module of a globular cluster.
        save : bool, optional
            Whether to save the obtained data with its information or not.
        **kwargs : additional optional arguments
        ra : float or str
            Right ascension coordinate for the centre of the scan (if no gc is provided).
        dec : float or str
            Declination coordinate for the centre of the scan (if no gc is provided)
        name : str
            String which provides the folder name where to save the data.
            Needed if no 'gc' object is supplied: if it is not given, and
            save was True, the data will be stored in the 'UntrackedData'
            folder.
        conditions : str or list of str
            List of conditions on the parameters to apply upon scanning the
            archive. If no conditions are supplied, no conditions are applied.
            Aliases:
            - 'cond'
            - 'conds'
            - 'condition'
            - 'conditions'

        Returns
        -------
        astro_cluster : astropy.Table
            Astropy table with  the query results.
        """
        astrometry = "source_id, ra, ra_error, dec, dec_error, parallax, parallax_error,\
              pmra, pmra_error, pmdec, pmdec_error"
        astro_sample = self.free_gc_query(radius, gc, save, data = astrometry, **kwargs)
        self._queryInfo['Scan Info']['Data Acquired'] = astrometry
        self._queryInfo['Flag']['Query'] = 'astrometry'
        return astro_sample

    def get_photometry(
        self,
        radius,
        gc: _Opt[_Union[_Cluster, str]] = None,
        save: str = False,
        **kwargs,
    ):
        """
        A pre-constructed ADQL search into the Gaia catalogue with fixed data
        retrieved, which is the principal photometric parameters, with the possibility
        to add personalized query conditions.

        The retrieved data is:
        'source_id, bp_rp, phot_bp_mean_flux, phot_rp_mean_flux, phot_g_mean_mag,
        phot_bp_rp_excess_factor, teff_gspphot'

        Parameters
        ----------
        radius : float
            Radius, in degrees, of the scan circle.
        gc : grasp.cluster.Cluster or str
            String name or Cluster object created with the G-GCAS module of a globular cluster.
        save : bool, optional
            Whether to save the obtained data with its information or not.
        **kwargs : additional optional arguments
        ra : float or str
            Right ascension coordinate for the centre of the scan (if no gc is provided).
        dec : float or str
            Declination coordinate for the centre of the scan (if no gc is provided)
        name : str
            String which provides the folder name where to save the data.
            Needed if no 'gc' object is supplied: if it is not given, and
            save was True, the data will be stored in the 'UntrackedData'
            folder.
        conditions : str or list of str
            Listo of conditions on the parameters to apply upon scanning the
            archive. If no conditions are supplied, no conditions are applied.
            Aliases:
            - 'cond'
            - 'conds'
            - 'condition'
            - 'conditions'

        Returns
        -------
        photo_cluster : astropy.Table
            Astropy table with the results.
        """
        
        photometry = "source_id, bp_rp, phot_bp_mean_flux, phot_rp_mean_flux, \
              phot_g_mean_mag, phot_bp_rp_excess_factor, teff_gspphot"
        phot_sample = self.free_gc_query(radius, gc, save, data = photometry, **kwargs)
        self._queryInfo['Scan Info']['Data Acquired'] = photometry
        self._queryInfo['Flag']['Query'] = 'photometry'
        return phot_sample

    def get_rv(
        self,
        radius,
        gc: _Opt[_Union[_Cluster, str]] = None,
        save: bool = False,
        **kwargs,
    ):
        """
        A pre-constructed ADQL search into the Gaia catalogue with fixed data
        retrieved, which is the radial velociti parameter with its error, with
        the possibility to add personalized query conditions.

        The retrieved data is:
        'source_id, radial_velocity, radial_velocity_error'

        Parameters
        ----------
        radius : float
            Radius, in degrees, of the scan circle.
        gc : grasp.cluster.Cluster or str
            String name or Cluster object created with the G-GCAS module of a globular cluster.
        save : bool, optional
            Whether to save the obtained data with its information or not.
        **kwargs : additional optional arguments
        ra : float or str
            Right ascension coordinate for the centre of the scan (if no gc is provided).
        dec : float or str
            Declination coordinate for the centre of the scan (if no gc is provided)
        name : str
            String which provides the folder name where to save the data.
            Needed if no 'gc' object is supplied: if it is not given, and
            save was True, the data will be stored in the 'UntrackedData'
            folder.
        conditions : str or list of str
            Listo of conditions on the parameters to apply upon scanning the
            archive. If no conditions are supplied, no conditions are applied.
            Aliases:
            - 'cond'
            - 'conds'
            - 'condition'
            - 'conditions'
        Returns
        -------
        rv_cluster : astropy.Table
            Astropy t able with te result.
        """
        
        rv = "source_id, radial_velocity, radial_velocity_error"
        conditions = get_kwargs(("cond", "conds", "conditions", "condition"), "None", kwargs)
        if conditions == "None":
            conditions = "radial_velocity IS NOT NULL"
        else: 
            conditions += ", radial_velocity IS NOT NULL"
        rv_sample = self.free_gc_query(radius, gc, save, data = rv, conditions=conditions, **kwargs)
        self._queryInfo['Scan Info']['Data Acquired'] = rv
        self._queryInfo['Flag']['Query'] = 'radial velocity'
        return rv_sample

    def _run_query(self, gc_id, ra, dec, radius, data, cond, save):
        """
        The actual sub-routine which sends the query, checking if data already
        exists with the same query conditions, in which case loading it.

        Parameters
        ----------
        gc_id : str
            Folder name, used to check de existing data and to save the qued data
            (if save=True). Usually is the Cluster's name.'
        ra : str
            Right ascension of the centre of the scan.
        dec : str
            Declination of the centre of the scan.
        radius : str
            Radius of the circular scan.
        data : str
            The data to retrieve from the query.
        conditions : list of str
            The conditions to apply on the requested data.
        save : bool
            Wether to save or not the retrieved data.
        """
        check = self.__check_query_exists(gc_id)
        if check is False:
            query = self._adqlWriter(ra, dec, radius, data=data, conditions=cond)
            job = _Gaia.launch_job_async(query)
            sample = job.get_results()
            print(f"Sample number of sources: {len(sample):d}")
            self.last_result = sample
            if save:
                self._saveQuery(sample, gc_id)
        else:
            print(
                f"""Found data with the same conditions for object {gc_id} in
{check[1]}.
Loading it..."""
            )
            sample = _loadData(name = check[1], as_sample=False)
            self.last_result = check[1]
            print(f"Sample number of sources: {len(sample):d}")
        return sample

    def _saveQuery(self, dat, name: str):
        """
        Routine for saving the query with its information, in the 'query_data.txt'
        and 'query_info.txt' files

        Parameters
        ----------
        dat : astropy.Table
            The astropy table containing all the retrieved data.
        name : str
            Where to save the data, usually the cluster's name.

        """
        config = _cp.ConfigParser()
        tn = _ts()
        if name.upper() == "UNTRACKEDDATA":
            fold = _UD
        else:
            fold = self._checkPathExist(name.upper())
        tnfold = _os.path.join(fold, tn)
        _os.mkdir(tnfold)
        data = _os.path.join(tnfold, _QDATA)
        info = _os.path.join(tnfold, _QINFO)
        if any([isinstance(dat, _Table), not isinstance(dat, _QTable)]):
            dat = _QTable(dat)
        dat = self._writeHeader(dat, name)
        dat.write(data, format="fits")
        for section, options in self._queryInfo.items():
            config[section] = options
        import warnings
        warnings.warn(
            "The 'query_info.ini' will be deprecated in future versions, shifting to the use of fits headers",
            DeprecationWarning,
        )
        with open(info, "w", encoding="UTF-8") as configfile:
            config.write(configfile)
        print(data)
        print(info)

    
    def _writeHeader(self, data: _QTable, name: str) -> _QTable:
        """
        Function to write the header of the query info file.

        Parameters
        ----------
        data : astropy.table.QTable
            The astropy table containing the data to save.
        name : str
            The name of the cluster.

        Returns
        -------
        info : dict
            The dictionary with the header added.

        """
        header = {
            "OBJECT": (
                name.upper() if not name.upper() == "UNTRACKEDDATA" else "UNDEF",
                'object name'),
            "RA": (self._queryInfo["Scan Info"]["RA"].value, 'right ascension of scan centre [deg]'),
            "DEC": (self._queryInfo["Scan Info"]["DEC"].value, 'declination of scan centre [deg]'),
            "RADIUS": (
                self._queryInfo["Scan Info"]["Scan Radius"], 'radius of scan circle [deg]',
            ),
            "CONDS": (
                not self._queryInfo["Scan Info"]["Conditions Applied"]=='None',
                'conditions applied to the query',
            )
        }
        if header["CONDS"] == True:
            for i, c in enumerate(self._queryInfo["Scan Info"]["Conditions Applied"]):
                header[f"COND{i}"] = c
        for key,value in header.items():
                data.meta[key] = value
        return data



    def _checkPathExist(self, dest: str):
        """
        Check if the path exists, and if not creates it.

        Parameters
        ----------
        dest : str
            the path to check.

        """
        self._fold = _CDF(dest)
        if not _os.path.exists(self._fold):
            _os.makedirs(self._fold)
            print(f"Path '{self._fold}' did not exist. Created.")
        return self._fold

    def _formatCheck(
        self, data: _Opt[_Union[str, list]], conditions: _Opt[_Union[str, list]]
    ):
        """
        Function to check and correct the format the 'data' and 'conditions'
        variables were imput with.

        Parameters
        ----------
        data : Optional[Union[str,list]]
            The data variable, containing all the requested parameters.
        conditions : Optional[Union[str,list]]
            The conditions variable, containing all the conditions to apply to the
            query.

        Returns
        -------
        dat : str
            The correct format for the data variable.
        cond : list of str
            The correct format for the conditions variable.

        """
        dat = ""
        cond = ""
        if data is not None:
            if isinstance(data, list):
                for i in range(len(data) - 1):
                    dat += data[i] + ", "
                dat += data[-1]
            else:
                dat = data
        else:
            dat = "source_id"
        if conditions != "None":
            if isinstance(conditions, str):
                conditions = conditions.split(",")
                cond = "AND "
                for i in range(len(conditions) - 1):
                    cond += (
                        conditions[i]
                        + """
    AND """
                    )
                cond += conditions[-1]
            else:
                cond = "AND "
                for i in range(len(conditions) - 1):
                    cond += (
                        conditions[i]
                        + """
    AND """
                    )
                cond += conditions[-1]
        return dat, cond

    def _adqlWriter(self, ra, dec, radius, data, conditions):
        """
        This function writes the query, correctly formatting all the variables
        in order to be accepted by the GAIA ADQL language.

        Parameters
        ----------
        ra : str
            Right ascension.
        dec : str
            Declination.
        radius : str
            Scan radius.
        data : str
            Data to retrieve.
        conditions : list of str
            Conditions to apply.

        Returns
        -------
        query : str
            The full string to input the query with.

        """
        if isinstance(ra, _u.Quantity):
            ra = ra / _u.deg
        if isinstance(dec, _u.Quantity):
            dec = dec / _u.deg
        if isinstance(radius, _u.Quantity):
            radius = radius / _u.deg
        circle = f"{ra},{dec},{radius:.3f}"
        dat, cond = self._formatCheck(data, conditions)
        query = self._baseQ.format(
            data=dat, table=self._table, circle=circle, cond=cond
        )
        self.last_query = query
        return query

    def _get_coordinates(self, gc, **kwargs):
        """
        Function to get the coordinates of the cluster, either from the Cluster
        object or from the kwargs.

        Parameters
        ----------
        gc : grasp.Cluste or str or None
            The cluster object or the string name of the cluster.
        **kwargs : dict
            The optional arguments to pass to the function.

        Returns
        -------
        ra : float
            Right ascension of the center of the scan.
        dec : float
            Declination of the center of the scan.
        savename : str
            NAme identifier of the objects, for the data path.

        """
        if gc is None:
            ra = kwargs.get("ra", None)
            dec = kwargs.get("dec", None)
            gc = _Cluster(ra=ra, dec=dec)
            savename = kwargs.get("name", "UntrackedData")
        else:
            if isinstance(gc, _Cluster):
                ra = gc.ra
                dec = gc.dec
                savename = gc.id
            elif isinstance(gc, str):
                gc = _Cluster(gc)
                ra = gc.ra
                dec = gc.dec
                savename = gc.id
        return gc, ra, dec, savename

    def __check_query_exists(self, name: str) -> bool | tuple[bool, str]:
        """
        Checks wether the requested query already exist saved for the Cluster.

        Parameters
        ----------
        name : str
            Folder where to search for saved data. Usually is the cluster's name

        Returns
        -------
        check : bool or tuple
            If no data was found to be compatible with the search, this is False.
            Either way it is a tuple, which first elemnt is True while the second
            is the complete file path to the corresponding saved data.
        """
        config = _cp.ConfigParser()
        try:
            tns = _tnlist(name)
        except FileNotFoundError:
            return False
        check = False
        for tn in tns:
            file_path = _os.path.join(tn, _QINFO)
            if _os.path.exists(file_path):
                config.read(file_path)
                try:
                    data_acquired = config["Scan Info"]["Data Acquired"]
                    conditions_applied = config["Scan Info"]["Conditions Applied"]
                    scan_radius = config["Scan Info"]["Scan Radius"]
                except KeyError as e:
                    print(f"Key error: {e}")
                    continue
                if (
                    data_acquired == self._queryInfo["Scan Info"]["Data Acquired"]
                    and conditions_applied
                    == self._queryInfo["Scan Info"]["Conditions Applied"]
                    and scan_radius == str(self._queryInfo["Scan Info"]["Scan Radius"])
                ):
                    check = (True, tn.split("/")[-1])
                    break
        return check

    def __load_table(self):
        """Loads the instanced table(s)"""
        if isinstance(self._table, list):
            table = _np.zeros(len(self._table), dtype=object)
            for i, t in enumerate(self._table):
                table[i] = _Gaia.load_table(t)
        else:
            table = _Gaia.load_table(self._table)
        return table


    def __descr(self) -> str:
        """Print out tables descriptions"""
        table = self.__load_table()
        text = ""
        if isinstance(table, _np.ndarray):
            for t in table:
                text += (
                    f"\n{t.name.upper()}\n" + "-" * len(t.name) + f"\n{t.description}\n"
                )
        else:
            text = (
                f"{table.name.upper()}\n"
                + "-" * len(table.name)
                + f"\n{table.description}"
            )
        return text


    def __params(self):
        """Print out available talbes parameters"""
        table = self.__load_table()
        text = ""
        if isinstance(table, _np.ndarray):
            for t in table:
                text += f"\n{t.name.upper()}\n" + "-" * len(t.name) + "\n"
                for column in t.columns:
                    text += f"{column.name}     "
                text += "\n"
        else:
            text = f"{table.name.upper()}\n" + "-" * len(table.name) + "\n"
            for column in table.columns:
                text += f"{column.name}     "
        return text


    def __get_repr(self):
        """Get text for '__repr__' method"""
        return "<grasp.query.GaiaQuery class>"


    def __get_str(self):
        """Get text for '__str__' method"""
        tables = self.__load_table()
        # the old reprt with table descriptions
        if isinstance(tables, _np.ndarray):
            text = ''
            for table in tables:
                text += (
                    f"\n{table.name.upper()}\n" + "-" * len(table.name) + f"\n{table.description}\n"
                )
                text += "\nParameters:\n"
                for column in table.columns:
                    text += f"{column.name}    "
                text += "\n"
        else:
            text = (
                f"\n{tables.name.upper()}\n"
                + "-" * len(tables.name)
                + f"\n{tables.description}\n"
            )
            text += "\nParameters:\n"
            for column in tables.columns:
                text += f"{column.name}     "
        return text
