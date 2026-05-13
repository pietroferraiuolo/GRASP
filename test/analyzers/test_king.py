import unittest

from grasp.analyzers.king import king_integrator


class TestKingFunctions(unittest.TestCase):

    def test_king_integrator(self):
        # This test assumes the Fortran90 code and required files are correctly set up.
        # It will not run correctly in this isolated environment.
        w0 = 5.0
        output = 'profile'
        try:
            result = king_integrator(w0, output)
            self.assertIsInstance(result, str)
        except Exception as e:
            self.fail(f"king_integrator raised an exception: {e}")
