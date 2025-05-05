"""
Author(s)
---------
    - Pietro Ferraiuolo : written in 2024

Description
-----------
This module contains R libraries implementation checks
for the GRASP package.
"""
from typing import Union
from logging import ERROR
from rpy2.robjects.packages import importr, isinstalled, LibraryError
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

# Disable unnecessary R logging
rpy2_logger.setLevel(ERROR)

utils = importr('utils')
utils.chooseCRANmirror(ind=1)


def check_packages(packages:Union[str, list[str]]) -> None:
    """
    Check if the R packages are installed.

    Parameters
    ----------
    packages : str or list[str]
        The name of the R package(s) to check.

    Returns
    -------
    None
    """
    if isinstance(packages, str):
        packages = [packages]
    for package in packages:
        if not isinstalled(package):
            print(f"Package '{package}' is not installed.\nInstalling it now...")
            utils.install_packages(package)
            print(f"Package '{package}' installed.")
        try:
            importr(package)
        except LibraryError as e:
            raise LibraryError(
                f"Package `{package}` installed but failed to import") from e
        print(f"Correctly imported `{package}`.")
            