import unittest
import numpy as np
import pandas as pd
from grasp import plots as gplt
from grasp.stats import fit_distribution as regression


class TestPlots(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.x = np.random.randn(100)
        self.y = np.random.randn(100)
        self.g = np.random.uniform(10, 20, 100)
        self.b_r = np.random.uniform(0, 2, 100)
        self.teff_gspphot = np.random.uniform(3000, 8000, 100)
        self.pmra = np.random.uniform(-5, 5, 100)
        self.pmdec = np.random.uniform(-5, 5, 100)
        self.ra = np.random.uniform(0, 360, 100)
        self.dec = np.random.uniform(-90, 90, 100)
        self.data = np.random.randn(100)
        self.dataerr = np.random.uniform(0.1, 0.5, 100)
        self.sample = pd.DataFrame(
            {
                "ra": self.ra,
                "dec": self.dec,
                "pmra": self.pmra,
                "pmdec": self.pmdec,
                "phot_g_mean_mag": self.g,
                "bp_rp": self.b_r,
                "teff_gspphot": self.teff_gspphot,
            }
        )
        self.data = np.random.randn(1000)

    def test_doubleHistScatter(self):
        try:
            gplt.doubleHistScatter(
                self.x,
                self.y,
                kde=True,
                kde_kind="gaussian",
                xlabel="X Axis",
                ylabel="Y Axis",
                title="doubleHistScatter",
            )
        except Exception as e:
            self.fail(f"doubleHistScatter raised an exception: {e}")

    def test_colorMagnitude(self):
        try:
            gplt.colorMagnitude(
                self.sample, alpha=0.5, cmap="viridis"
            )
        except Exception as e:
            self.fail(f"colorMagnitude raised an exception: {e}")

    def test_properMotion(self):
        try:
            gplt.properMotion(self.sample, color="blue", s=10, alpha=0.7)
        except Exception as e:
            self.fail(f"properMotion raised an exception: {e}")

    def test_spatial(self):
        try:
            gplt.spatial(self.sample, color="red", s=15, alpha=0.6)
        except Exception as e:
            self.fail(f"spatial raised an exception: {e}")

    def test_histogram(self):
        try:
            result = gplt.histogram(
                self.data, kde=True, kde_kind="gaussian", xlabel="Data", out=True
            )
            self.assertIn("h", result)
            self.assertIn("kde", result)
        except Exception as e:
            self.fail(f"histogram raised an exception: {e}")

    def test_scatXHist(self):
        try:
            result = gplt.scatterXHist(
                self.x, self.y, xerr=0.1, xlabel="X Axis", ylabel="Y Axis"
            )
            self.assertEqual(len(result), 2)
        except Exception as e:
            self.fail(f"scatXHist raised an exception: {e}")

    def test_errorbar(self):
        try:
            gplt.errorbar(
                self.y,
                np.abs(self.y * 0.1435),
                x=self.x,
                xerr=0.1,
                fmt="o",
                color="green",
                ecolor="black",
                xlabel="X Axis",
                ylabel="Y Axis",
                title="Errorbar Plot",
            )
        except Exception as e:
            self.fail(f"errorbar raised an exception: {e}")

    def test_regressionPlot(self):
        self.regression_model = regression(self.data, kind="gaussian")
        try:
            gplt.regressionPlot(
                self.regression_model, xlabel="X Axis", title="Regression Plot"
            )
        except Exception as e:
            self.fail(f"regression raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
