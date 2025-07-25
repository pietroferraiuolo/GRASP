"""
Author(s)
---------
- Pietro Ferraiuolo : Written in 2024

Description
-----------

How to Use
----------

Examples
--------

"""

import os as _os
import datetime as _dt
from typing import Any as _Any, Optional as _opt
from grasp.core import folder_paths as _fn
from astropy.table import QTable as _QTable

datapath = _fn.BASE_DATA_PATH
querypath = _fn.QUERY_DATA_FOLDER


def load_data(
    *,
    filepath: _opt[str] = None,
    tn: _opt[str] = None,
    data_format: str = "fits",
    file_format: str = ".fits",
    as_sample: bool = True,
    **kwargs: _Any,
) -> _QTable | _Any:
    """
    Loads the data from a file as an Astropy quantity table.

    Parameters
    ----------
    filepath : str, optional
        The complete path to the file to load. This can be omitted if tn is provided.
    data_format : str, optional
        The format of the file to load. Default is 'fits'. Refer to the Astropy
        QTable.read documentation for supported formats.
    file_format : str, optional
        The file extension to use when constructing the file name. Default is '.fits'.
    as_sample : bool, optional
        If True, the loaded data will be returned as a Sample object; otherwise,
        it will be returned as a QTable.

    Additional Parameters
    ---------------------
    tn : str, optional
        Tracking number of the data to load. If provided, the function will search
        for the file in the corresponding tracking number folder.

    Returns
    -------
    data : astropy.table.QTable or grasp.Sample
        The loaded data from the file. If `as_sample` is True, the data is returned
        as a Sample object; otherwise, it is returned as a QTable.

    Raises
    ------
    FileNotFoundError
        If the specified file cannot be found.
    ValueError
        If both `tn` and `name` are None.

    Examples
    --------
    Load data using a tracking number:
        >>> data = load_data(tn="20240101_123456")

    Load data using a specific file path:
        >>> data = load_data(name="/path/to/your_data.fits")

    Load data as a QTable instead of a Sample:
        >>> data = load_data(tn="20240101_123456", as_sample=False)
    """
    tn = kwargs.get('tn', None)
    if tn is not None:
        file_name = (
            ("query_data" + file_format) if filepath is None else (filepath + file_format)
        )
        file_path = _findTracknum(tn, complete_path=True)
        file = _os.path.join(file_path, file_name)
    else:
        file = filepath
    data = _QTable.read(file, format=data_format)
    if as_sample:
        from grasp._utility.sample import Sample, GcSample

        if file_format == ".fits" and "OBJECT" in data.meta.keys():
            gc = data.meta["OBJECT"]
            if gc == "UNDEF":
                gc = "UntrackedData"
        else:
            try:
                from grasp import Cluster

                gc = _os.path.dirname(file).split("/")[-2]
                gc = Cluster(gc)
            except Exception:
                gc = "UntrackedData"
        if gc == "UntrackedData":
            data = Sample(data)
        else:
            data = Sample(data, gc=gc)
    return data


def get_file_list(tn: str = None, fold: str = None, key: str = None) -> str | list[str]:
    """
    Returns the file list of a given globular cluster datapath.

    Parameters
    ----------
    tn: str
        Tracking number of the data to search for.
    fold : str
        The name of the folder to search for the tracking number, or the complete
        folder path, including the tn, in case it is a data folder (then 'tn' must
        be none, as default).
    key : str, optional
        A key which identify specific files to return.

    Returns
    -------
    fl : list os str
        List of sorted files inside the folder.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    TypeError
        If the 'key' argument is not a string.

    Examples
    --------
    Here are some examples regarding the use of the 'key' argument. Let's say
    we need a list of files inside ''tn = '20160516_114916' '' for GC 'ngc104'

        >>> tn = '20160516_114916'
        >>> get_file_list(tn=tn) # if only that [gc_name] folder has that tn folder
        ['.../GRASP/grasp/data/query/[gc_name]/[tn]/query_data.txt',
         '.../GRASP/grasp/data/query/[gc_name]/[tn]/spatial_data.txt',
         '.../GRASP/grasp/data/query/[gc_name]/[tn]/velocity_data.txt',
         '.../GRASP/grasp/data/query/[gc_name]/[tn]/query_info.ini',
         '.../GRASP/grasp/data/query/[gc_name]/[tn]/dynamical_data.txt']

    Let's suppose we want only the list of 'xxx_data.txt' files:

        >>> get_file_list(tn=tn, key='_data')
        ['.../GRASP/grasp/data/query/[gc_name]/[tn]/query_data.txt',
         '.../GRASP/grasp/data/query/[gc_name]/[tn]/spatial_data.txt',
         '.../GRASP/grasp/data/query/[gc_name]/[tn]/velocity_data.txt',
         '.../GRASP/grasp/data/query/[gc_name]/[tn]/dynamical_data.txt']

    This function can be used to retrieve a list of folders too. Say we want to
    know which cluster has king-model data available:

        >>> foldpath = '.../GRASP/grasp/data/models/'
        >>> fold_list = get_file_list(fold=foldpath)
        >>> fold_list
        ['.../GRASP/grasp/data/models/NGC104',
         '.../GRASP/grasp/data/models/NGC4372',
         '.../GRASP/grasp/data/models/NGC6121']

    These are the folders of the clusters which have king-model data available. indeed,
    if we check:

        >>> get_file_list(fold=fold_list[0])
        ['.../GRASP/grasp/data/models/NGC104/SM_king.txt']
    """
    if tn is None and fold is not None:
        fl = sorted([_os.path.join(fold, file) for file in _os.listdir(fold)])
    else:
        if fold is None:
            fold = _findTracknum(tn, complete_path=True)
            if len(fold) == 0:
                raise FileNotFoundError(f"No data found with '{tn = }'")
            fl = sorted([_os.path.join(fold, file) for file in _os.listdir(fold)])
        else:
            try:
                paths = _findTracknum(tn, complete_path=True)
                if isinstance(paths, str):
                    paths = [paths]
                for path in paths:
                    if fold in path.split("/")[-2]:
                        fl = sorted(
                            [_os.path.join(path, file) for file in _os.listdir(path)]
                        )
                    else:
                        raise Exception
            except Exception as exc:
                raise FileNotFoundError(
                    f"Invalid Path: no data found for '.../{fold}/{tn}'"
                ) from exc
    if key is not None:
        try:
            selected_list = []
            for file in fl:
                if key in file.split("/")[-1]:
                    selected_list.append(file)
        except TypeError as err:
            raise TypeError("'key' argument must be a string") from err
        fl = selected_list
    if len(fl) == 1:
        fl = fl[0]
    return fl


def get_kwargs(names: tuple[str], default: _Any, kwargs: dict[str, _Any]) -> _Any:
    """
    Gets a tuple of possible kwargs names for a variable and checks if it was
    passed, and in case returns it.

    Parameters
    ----------
    names : tuple
        Tuple containing all the possible names of a variable which can be passed
        as a **kwargs argument.
    default : any type
        The default value to assign the requested key if it doesn't exist.
    kwargs : dict
        The dictionary of variables passed as 'Other Parameters'.

    Returns
    -------
    key : value of the key
        The value of the searched key if it exists. If not, the default value will
        be returned.
    """
    possible_keys = names
    for key in possible_keys:
        if key in kwargs:
            return kwargs[key]
    return default


def tnlist(gc_name: str):
    """
    Returns the list of tracking numbers for a given globular cluster.

    Parameters
    ----------
    gc_name : str
        Name of the globular cluster.

    Returns
    -------
    tns : list of str
        List of tracking numbers for the given globular cluster.

    """
    basepath = _fn.CLUSTER_DATA_FOLDER(gc_name)
    tn = _os.listdir(basepath)
    tns = sorted([_os.path.join(basepath, tt) for tt in tn])
    return tns


def timestamp():
    """
    Creates a new tracking number in the format 'yyyymmdd_HHMMSS'

    Returns
    -------
    tn : str
        Tracking number.

    """
    tn = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return tn


def _findTracknum(tn: str, complete_path: bool = True) -> str | list[str]:
    """
    Search for the tracking number given in input within all the data path subfolders.

    Parameters
    ----------
    tn : str
        Tracking number to be searched.
    complete_path : bool, optional
        Option for wheter to return the list of full paths to the folders which
        contain the tracking number or only their names.

    Returns
    -------
    tn_path : list of str
        List containing all the folders (within the OPTData path) in which the
        tracking number is present, sorted in alphabetical order.

    """
    tn_path: list[str] = []
    for data_type in _os.listdir(datapath):
        querypath = _os.path.join(datapath, data_type)
        for fold in _os.listdir(querypath):
            search_fold = _os.path.join(querypath, fold)
            if tn in _os.listdir(search_fold):
                if complete_path:
                    tn_path.append(_os.path.join(search_fold, tn))
                else:
                    tn_path.append(fold)
    path_list = sorted(tn_path)
    if len(path_list) == 1:
        path_list = path_list[0]
    return path_list
