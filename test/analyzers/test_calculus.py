import unittest
import numpy as np
import sympy as sp
from grasp.analyzers.calculus import (
    compute_numerical_function, 
    compute_error, 
    gaus_legendre_integrator
)

class TestCalculusFunctions(unittest.TestCase):

    def test_compute_numerical_function(self):
        x = sp.symbols('x')
        func = x**2
        variables = [x]
        var_data = [np.array([1, 2, 3])]
        result = compute_numerical_function(func, variables, var_data)
        expected = np.array([1, 4, 9])
        np.testing.assert_equal(result.shape, expected.shape)
        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_error(self):
        x, y, e_x, e_y = sp.symbols('x y e_x e_y')
        func = x + y
        variables = [x, y, e_x, e_y]
        var_data = [np.array([1, 2]), np.array([3, 4])]
        var_errors = [np.array([0.1, 0.1]), np.array([0.1, 0.1])]
        result = compute_error(func, variables, var_data, var_errors)
        expected = np.array([4., 6.])
        np.testing.assert_array_almost_equal(result, expected)

    def test_gaus_legendre_integrator(self):
        def f(x):
            return x**2
        a = 0
        b = 1
        points = 20
        result = gaus_legendre_integrator(f, a, b, points)
        expected = 1/3  # Integral of x^2 from 0 to 1
        self.assertAlmostEqual(result, expected, places=5)


if __name__ == '__main__':
    unittest.main()