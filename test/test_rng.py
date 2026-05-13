"""Tests for the centralised RNG helper (P3-4)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from grasp.stats import XD_estimator, gaussian_mixture_model, kfold_gmm_estimator
from grasp.utils.rng import default_rng


class TestDefaultRng(unittest.TestCase):
    """``grasp.utils.rng.default_rng`` should wrap ``np.random.default_rng``."""

    def test_returns_generator(self) -> None:
        rng = default_rng()
        self.assertIsInstance(rng, np.random.Generator)

    def test_seed_reproducibility(self) -> None:
        a = default_rng(seed=12345).standard_normal(size=10)
        b = default_rng(seed=12345).standard_normal(size=10)
        np.testing.assert_array_equal(a, b)

    def test_no_seed_is_nondeterministic(self) -> None:
        a = default_rng().standard_normal(size=10)
        b = default_rng().standard_normal(size=10)
        # Vanishingly unlikely two independent draws coincide.
        self.assertFalse(np.array_equal(a, b))


class TestStochasticAPIRespectsSeed(unittest.TestCase):
    """Calling stochastic public APIs with the same ``seed`` must reproduce."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.data = np.vstack(
            [rng.normal(0, 1, size=(150, 2)), rng.normal(5, 1, size=(150, 2))]
        )

    def test_gaussian_mixture_model_is_reproducible(self) -> None:
        a = gaussian_mixture_model(self.data, n_components=2, seed=42)
        b = gaussian_mixture_model(self.data, n_components=2, seed=42)
        np.testing.assert_allclose(np.sort(a.means.ravel()), np.sort(b.means.ravel()))
        self.assertAlmostEqual(a.bic, b.bic, places=6)

    def test_kfold_gmm_is_reproducible(self) -> None:
        a = kfold_gmm_estimator(self.data, folds=4, n_components=2, seed=11)
        b = kfold_gmm_estimator(self.data, folds=4, n_components=2, seed=11)
        self.assertAlmostEqual(a.mean_loglik, b.mean_loglik, places=6)
        self.assertAlmostEqual(a.mean_bic, b.mean_bic, places=6)

    def test_xd_estimator_seed_is_honoured(self) -> None:
        try:
            import astroML.density_estimation  # noqa: F401
        except ImportError:
            self.skipTest("astroML not installed; XDGMM unavailable")
        x = self.data
        e = np.full_like(x, 0.05)
        m1 = XD_estimator(x, e, n_components=2, seed=7)
        m2 = XD_estimator(x, e, n_components=2, seed=7)
        np.testing.assert_allclose(np.sort(m1.mu.ravel()), np.sort(m2.mu.ravel()))


class TestEnvVarBackendToggle(unittest.TestCase):
    """``GRASP_R_BACKEND=1`` should be honoured even when no kwarg is passed."""

    def test_env_var_truthy(self) -> None:
        from grasp.stats import _use_r_backend

        os.environ.pop("GRASP_R_BACKEND", None)
        self.assertFalse(_use_r_backend(None))
        os.environ["GRASP_R_BACKEND"] = "1"
        try:
            self.assertTrue(_use_r_backend(None))
        finally:
            del os.environ["GRASP_R_BACKEND"]

    def test_explicit_python_overrides_env(self) -> None:
        from grasp.stats import _use_r_backend

        os.environ["GRASP_R_BACKEND"] = "1"
        try:
            self.assertFalse(_use_r_backend("python"))
        finally:
            del os.environ["GRASP_R_BACKEND"]


class TestPyModelSerialization(unittest.TestCase):
    """``PyFitResult.save/load`` should round-trip via joblib (P3-5)."""

    def test_gaussian_fit_roundtrip(self) -> None:
        from grasp.analyzers.backends._python import fit_distribution_python

        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, size=2000)
        fit = fit_distribution_python(x, method="gaussian")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fit.joblib")
            fit.save(path)
            from grasp.analyzers.backends._python import PyFitResult

            loaded = PyFitResult.load(path)
        np.testing.assert_allclose(fit.coeffs, loaded.coeffs)

    def test_gmm_roundtrip(self) -> None:
        from grasp.analyzers.backends._python import (
            GaussianMixturePython,
            gaussian_mixture_model_python,
        )

        rng = np.random.default_rng(0)
        data = np.vstack(
            [rng.normal(0, 1, (100, 2)), rng.normal(5, 1, (100, 2))]
        )
        gmm = gaussian_mixture_model_python(data, n_components=2, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gmm.joblib")
            gmm.save(path)
            loaded = GaussianMixturePython.load(path)
        np.testing.assert_allclose(np.sort(gmm.means.ravel()), np.sort(loaded.means.ravel()))


if __name__ == "__main__":
    unittest.main()
