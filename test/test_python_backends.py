"""Tests for the Python-native backends introduced in Phase 3.

These cover:

* :func:`grasp.analyzers.backends._python.fit_distribution_python` (P3-1).
* :func:`grasp.analyzers.backends._python.linear_regression_python` (P3-3).
* :class:`grasp.analyzers.backends._python.GaussianMixturePython` (P3-2).
* :func:`grasp.analyzers.backends._python.kfold_gmm_python` (P3-2).

R parity tests are *skipped* automatically when :mod:`rpy2` is unavailable
(see ``r_check.py``); they are kept here as smoke tests rather than as
authoritative agreement checks, because R is deprecated and being phased out.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from grasp.analyzers.backends._python import (
    fit_distribution_python,
    gaussian_mixture_model_python,
    kfold_gmm_python,
    linear_regression_python,
)


class TestFitDistributionPython(unittest.TestCase):

    def test_gaussian_recovers_mu_sigma(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(loc=1.2, scale=0.7, size=10000)
        fit = fit_distribution_python(x, method="gaussian")
        # coeffs = (A, mu, sigma) — check the location/scale agree with the
        # sample statistics to within a few percent.
        _, mu_hat, sigma_hat = fit.coeffs
        self.assertAlmostEqual(mu_hat, 1.2, delta=0.05)
        self.assertAlmostEqual(abs(sigma_hat), 0.7, delta=0.05)

    def test_exponential_uses_decay_form(self) -> None:
        """Phase 1 (P1-7): the Python model must be ``A * exp(-b * x)``."""

        rng = np.random.default_rng(0)
        scale = 2.5
        x = rng.exponential(scale=scale, size=20000)
        fit = fit_distribution_python(x, method="exponential")
        _, b_hat = fit.coeffs
        # The PDF of an Exp(scale) is (1/scale) * exp(-x/scale), so the rate
        # parameter b should approach 1/scale.
        self.assertAlmostEqual(b_hat, 1.0 / scale, delta=0.1)
        self.assertGreater(b_hat, 0.0, msg="Decay form requires positive b")

    def test_invalid_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_distribution_python(np.linspace(0, 1, 100), method="not_a_model")


class TestLinearRegressionPython(unittest.TestCase):

    def test_recovers_known_line(self) -> None:
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 200)
        y = 3.0 + 2.0 * x + rng.normal(0, 0.01, size=x.size)
        out = linear_regression_python(x, y)
        intercept, slope = out["coeffs"]
        self.assertAlmostEqual(intercept, 3.0, delta=1e-2)
        self.assertAlmostEqual(slope, 2.0, delta=1e-3)
        self.assertGreater(out["rsquared"], 0.999)
        self.assertEqual(len(out["p_values"]), 2)


class TestGaussianMixturePython(unittest.TestCase):

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.data = np.vstack(
            [rng.normal(0, 1, (200, 2)), rng.normal(6, 1, (200, 2))]
        )

    def test_recovers_two_clusters(self) -> None:
        gmm = gaussian_mixture_model_python(self.data, n_components=2, seed=0)
        means_sorted = np.sort(gmm.means[:, 0])
        np.testing.assert_allclose(means_sorted, [0.0, 6.0], atol=0.3)
        self.assertGreater(gmm.loglik, -np.inf)
        self.assertGreater(gmm.bic, 0)

    def test_unknown_mclust_name_warns(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gaussian_mixture_model_python(
                self.data, n_components=2, model_name="ZZZ", seed=0
            )
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("Unrecognised mclust" in m for m in messages))


class TestKFoldGMMPython(unittest.TestCase):

    def test_two_cluster_data_prefers_two_components(self) -> None:
        rng = np.random.default_rng(0)
        data = np.vstack(
            [rng.normal(0, 1, (200, 2)), rng.normal(8, 1, (200, 2))]
        )
        result_k1 = kfold_gmm_python(data, folds=4, n_components=1, seed=0)
        result_k2 = kfold_gmm_python(data, folds=4, n_components=2, seed=0)
        self.assertLess(
            result_k2.mean_bic,
            result_k1.mean_bic,
            msg="2-component GMM should beat 1-component on bimodal data.",
        )


if __name__ == "__main__":
    unittest.main()
