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

import astropy.units as u
import numpy as np


def velocity_conversion(mu, gc_distance, mu_error=0, gc_distance_error=0):
    """
    Converts the proper motion into velocities in km/s, with its error if provided.

    Parameters
    ----------
    mu : float or ArrayLike
        The proper motion component to convert.
    gc_distance : float
        The reference distance to use to make the conversion.
    mu_error : float or ArrayLike, optional
        The error(s) associated to the Proper Motion data. The default is 0,
        which means 'not provided'.
    gc_distance_error : float, optional
        The error associated to the reference distance. The default is 0, which
        means 'not provided'.

    Returns
    -------
    vkms : float or ArrayLike
        The converted velocity in km/s.
    vkms_err : float or ArrayLike
        The related, converted, error.

    """
    vkms = mu.to(u.mas / u.yr).to(u.rad / u.s) * gc_distance.to(u.kpc).to(u.km) / u.rad
    vkms_err = (
        np.sqrt(
            gc_distance**2 * mu_error.to(u.rad / u.s) ** 2
            + mu.to(u.rad / u.s) ** 2 * gc_distance_error**2
        )
        / u.rad
    )
    return vkms, vkms_err


# Passare il Sample object, così da far fare tutto a lui


def compute_r2d():

    return


def compute_r3d():

    return


def compute_Vtot():

    return


#
def cartesian_conversion(ra, dec):

    return


def polar_conversion(x, y):

    return


def cartesian_pm(pmra, pmdec, ra, dec):

    return


def polar_pm(mux, muy, ra, dec):

    return


def density_profile(data):
    """
    Returns the *observed* radial density profile for the given sample

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    rho : TYPE
        DESCRIPTION.

    """
    n_bin = int(1.5 * len(data) ** 0.5)
    dh = np.histogram(data, bins=n_bin)
    V = np.zeros(n_bin)
    rr1 = np.zeros(n_bin + 1)
    rr2 = np.zeros(n_bin + 1)
    bw = data.max() / n_bin
    rr1[0] = 0.0
    rr2[0] = bw
    for x in range(0, n_bin):
        V[x] = (4 / 3) * np.pi * (rr2[x] ** 3 - rr1[x] ** 3)
        rr1[x + 1] = rr1[x] + bw
        rr2[x + 1] = rr2[x] + bw
    rho = dh[0] / (V**3)
    return rho
