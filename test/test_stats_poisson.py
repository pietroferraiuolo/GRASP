"""Tests for the new Python Poisson MLE replacement (P1-8)."""

import unittest

import numpy as np

from grasp.stats import fit_poisson_mle


class TestPoissonMLE(unittest.TestCase):
    def test_mle_recovers_lambda(self):
        rng = np.random.default_rng(20260513)
        lam_true = 4.7
        counts = rng.poisson(lam=lam_true, size=5000)
        model = fit_poisson_mle(counts)
        a, lam_hat = model.coeffs
        self.assertAlmostEqual(a, counts.size, places=5)
        self.assertAlmostEqual(lam_hat, lam_true, delta=0.1)

    def test_mle_matches_sample_mean(self):
        """For a Poisson, the MLE of lambda is the sample mean."""
        rng = np.random.default_rng(20260514)
        counts = rng.poisson(lam=2.0, size=200)
        model = fit_poisson_mle(counts)
        _, lam_hat = model.coeffs
        np.testing.assert_allclose(lam_hat, counts.mean(), rtol=1e-4, atol=1e-5)

    def test_rejects_non_integer(self):
        with self.assertRaises(ValueError):
            fit_poisson_mle(np.array([0.5, 1.5, 2.5]))

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            fit_poisson_mle(np.array([-1, 0, 1]))

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            fit_poisson_mle(np.array([], dtype=int))


if __name__ == "__main__":
    unittest.main()
