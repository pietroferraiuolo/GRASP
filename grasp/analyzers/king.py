"""
:module: grasp.analyzers.king
:synopsis: This module provides a function to call the Fortran90 code for the
    Single-Mass King model integration routine.

Author(s)
---------
- Pietro Ferraiuolo : Written in 2024

Description
-----------
This module provides a function to call the Fortran90 code for the Single-Mass
King model integration routine.

Functions
---------
- king_integrator : Calls the Fortran90 code for the Single-Mass King model
    integration routine.

"""
import os
import subprocess
from grasp.core.osutils import get_file_list
from grasp.core.folder_paths import KING_INTEGRATOR_FOLDER
_king_dir = KING_INTEGRATOR_FOLDER
_king_exe = os.path.join(_king_dir, 'king_integrator')


def king_integrator(w0: float | int | str, output: str | list[str] = 'profile') -> str | list[str]:
    r"""
    This function calls a Fortran90 code for the Single-Mass King model 
    integration routine.

    Taking as imput a value for the King $W_0$ parameter, it will perform the
    integration of the model, producing a series of output data files, described
    in the 'Returns' section.

    Parameters
    ----------
    w0 : float
        King w0 parameter, which is the central potential well.
    output : str, optional
        Specifies which output file(s) to retain. The default is 'profile'.
        Options :
        - all : All of the below produced files
        - CalCurve : Caloric curve of the system
        - Cv :
        - CvNtK :
        - Er :
        - Etot : total energy of the system
        - params :
        - phi : information about the gravitational potential of the system
        - profiles : (normalized) density and w0 profiles with respect to the dimentionless radial distance from the centre
        - Skin : Surface kinetick energy distribution
        - x0Cv :

    Returns
    -------
    result : str or list
        The full path of the selected output file(s) as string
        (or list of strings if multiple output files have been selected).
    """
    if isinstance(w0, (float, int)):
        w0 = str(w0)
    result = subprocess.run([_king_exe, w0], capture_output=True, text=True,
                            cwd=_king_dir, check=False)
    if result.returncode != 0:
        print("Error during trhe Fortran90 code execution:")
        print(result.stderr)
    else:
        print("Calling Fortran90 code executor...\n")
        print(result.stdout)
    filelist = get_file_list(fold=_king_dir, key='.dat')
    result = []
    if 'all' in output:
        result = filelist
    elif isinstance(output, list):
        for entry in output:
            for i,file in enumerate(filelist):
                if entry in filelist:
                    result.append(file)
                    filelist.pop(i)
    else:
        for i,file in enumerate(filelist):
            if output in file:
                result = file
                filelist.pop(i)
    for file in filelist:
        os.remove(file)
    return result