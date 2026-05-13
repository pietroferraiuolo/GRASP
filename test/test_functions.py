"""Tests for :mod:`grasp.functions`."""

import math
import unittest

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from grasp.functions import CartesianConversion


class TestCartesianConversionUnits(unittest.TestCase):
    """P1-3: cross-check the tangent-plane projection against astropy.

    The class uses the projection

        x = sin(alpha - alpha0) cos(delta0)
        y = sin(delta) cos(delta0) - cos(delta) sin(delta0) cos(alpha - alpha0)
        r = sqrt(x**2 + y**2)

    which differs from the standard orthographic projection by ``cos(delta0)``
    vs ``cos(delta)`` in ``x``. For small angular separations these tests
    therefore allow a ``2%`` relative tolerance against the true angular
    separation (computed by :meth:`astropy.coordinates.SkyCoord.separation`).
    The point of the test is that the *units* are correct: if degrees were
    treated as radians the result would be ``57.3x`` larger and the
    assertions would fail catastrophically.
    """

    def setUp(self):
        # Mix of small and slightly larger separations across the sky.
        self.pairs = [
            (10.0, 20.0, 10.1, 20.05),
            (45.0, -30.0, 45.05, -29.95),
            (300.0, 60.0, 300.2, 60.1),
        ]

    def _astropy_separation_rad(self, ra0, dec0, ra, dec):
        c0 = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg)
        c1 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        return c0.separation(c1).to(u.rad).value

    def test_radial_distance_matches_skycoord_separation(self):
        for ra0, dec0, ra, dec in self.pairs:
            with self.subTest(ra0=ra0, dec0=dec0, ra=ra, dec=dec):
                cc = CartesianConversion(ra0=ra0, dec0=dec0, propagate_error=False)
                cc.compute([np.array([ra]), np.array([dec])])
                r_tangent = float(cc._values["r"][0])
                ref = self._astropy_separation_rad(ra0, dec0, ra, dec)
                # 2% rtol absorbs the O(delta - delta0)**2 mismatch between the
                # in-house projection and the true angular separation; this is
                # still ~30x tighter than would be allowed if degrees were
                # being used in place of radians.
                np.testing.assert_allclose(r_tangent, ref, rtol=2e-2, atol=1e-9)
                # And a hard sanity check that we are *not* mis-treating degrees
                # as radians (which would push r to roughly 57x the expected
                # value):
                self.assertLess(r_tangent, ref * 5.0)

    def test_radians_input(self):
        """Passing radians explicitly should match the default deg path."""
        ra0_deg, dec0_deg = 10.0, 20.0
        ra_deg, dec_deg = 11.0, 21.0

        cc_deg = CartesianConversion(ra0=ra0_deg, dec0=dec0_deg, propagate_error=False)
        cc_deg.compute([np.array([ra_deg]), np.array([dec_deg])])

        ra0_rad = math.radians(ra0_deg)
        dec0_rad = math.radians(dec0_deg)
        ra_rad = math.radians(ra_deg)
        dec_rad = math.radians(dec_deg)
        cc_rad = CartesianConversion(
            ra0=ra0_rad, dec0=dec0_rad, propagate_error=False, input_unit="rad"
        )
        cc_rad.compute([np.array([ra_rad]), np.array([dec_rad])])

        for name in ("x", "y", "r", "theta"):
            np.testing.assert_allclose(
                float(cc_deg._values[name][0]),
                float(cc_rad._values[name][0]),
                rtol=1e-10,
                atol=1e-12,
            )

    def test_quantity_input(self):
        """``astropy.units.Quantity`` inputs should be auto-converted."""
        cc_q = CartesianConversion(
            ra0=10.0 * u.deg, dec0=20.0 * u.deg, propagate_error=False
        )
        cc_q.compute([np.array([11.0]) * u.deg, np.array([21.0]) * u.deg])

        cc_deg = CartesianConversion(ra0=10.0, dec0=20.0, propagate_error=False)
        cc_deg.compute([np.array([11.0]), np.array([21.0])])

        np.testing.assert_allclose(
            float(cc_q._values["r"][0]),
            float(cc_deg._values["r"][0]),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_invalid_input_unit(self):
        with self.assertRaises(ValueError):
            CartesianConversion(ra0=10.0, dec0=20.0, input_unit="arcsec")


if __name__ == "__main__":
    unittest.main()
