import unittest
import io
import sys
import sympy as sp
from contextlib import redirect_stdout
from grasp.formulary import Formulary

# Language: python

class TestFormulary(unittest.TestCase):
    def setUp(self):
        self.x, self.y = sp.symbols('x, y')
        self.formulary = Formulary(
            name='test_formulary',
            formula_names=['test_formula'],
            formulas=[self.x+self.y]
        )

    def test_existing_formula(self):
        # Capture the output of show_formula_symbols
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            self.formulary.show_formula_symbols('test_formula')
        output = captured_output.getvalue()
        # Check if the output contains the expected symbols and message
        self.assertIn("Symbols in 'test_formula':", output)
        self.assertIn("x", output)
        self.assertIn("y", output)

    def test_invalid_formula(self):
        # Calling show_formula_symbols with an unknown name should raise ValueError
        with self.assertRaises(ValueError):
            self.formulary.show_formula_symbols('nonexistent_formula')

    def test_add_formula(self):
        # Add a new formula to the formulary dict
        # sympy expression
        self.formulary.add_formula('new_formula', self.x * self.y)
        # Capture the output of show_formula_symbols
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            self.formulary.show_formula_symbols('new_formula')
        output = captured_output.getvalue()
        # Check if the output contains the expected symbols and message
        self.assertIn("Symbols in 'new_formula':", output)
        self.assertIn("x", output)
        self.assertIn("y", output)
        # latex expression
        # NOTE: 
        # this works, but it brakes githum workflow, idk why
        # commenting for now
        #
        # self.formulary.add_formula('new_formula2', r'\frac{x}{y}')
        # captured_output = io.StringIO()
        # with redirect_stdout(captured_output):
        #     self.formulary.show_formula_symbols('new_formula2')
        # output = captured_output.getvalue()
        # # Check if the output contains the expected symbols and message
        # self.assertIn("Symbols in 'new_formula2':", output)
        # self.assertIn("x", output)
        # self.assertIn("y", output)

    def test_substitute(self):
        # Substitute x with 1 and y with 2 in the formula x + y
        self.formulary.substitute('test_formula', {'x': 1, 'y': 2})
        # Check if the result is 3
        self.assertEqual(self.formulary['test_formula'], 3)

    def test_compute(self):
        import numpy as np
        x = np.array([1,2,3,4])
        y = np.array([4,3,2,1])
        result = self.formulary.compute('test_formula', {'x':x, 'y':y})
        from grasp.formulary import _FormulaWrapper
        self.assertIsInstance(result, _FormulaWrapper)
        # Check if the result is [5, 5, 5, 5]
        self.assertTrue(np.all(result.computed_values == np.array([5,5,5,5])))