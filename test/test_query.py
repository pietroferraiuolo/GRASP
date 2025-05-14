import unittest
from unittest.mock import patch, MagicMock
from grasp.gaia.query import GaiaQuery
from grasp._utility.cluster import Cluster
from astropy.table import QTable
from pandas import DataFrame

class TestGaiaQuery(unittest.TestCase):

    def setUp(self):
        self.gaia_query = GaiaQuery()

    def test_initialization(self):
        self.assertEqual(self.gaia_query._table, "gaiadr3.gaia_source")

    @patch('grasp.gaia.query.GaiaQuery._run_query')
    def test_free_gc_query(self, mock_run_query):
        mock_run_query.return_value = QTable({'col1': [1, 2], 'col2': [3, 4]})
        cluster = Cluster('ngc104')
        result = self.gaia_query.free_gc_query(0.1, gc=cluster)
        self.assertIsNotNone(result)
        mock_run_query.assert_called_once()

    @patch('grasp.gaia.query.GaiaQuery._run_query')
    def test_free_query(self, mock_run_query):
        mock_run_query.return_value = DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        cluster = Cluster('ngc104')
        query_cmd = """SELECT * FROM gaiadr3.gaia_source WHERE ra BETWEEN 0 AND 1 AND dec BETWEEN 0 AND 1"""
        result = self.gaia_query.free_gc_query(query_cmd)
        self.assertIsNotNone(result)
        mock_run_query.assert_called_once()

    @patch('grasp.gaia.query.GaiaQuery._run_query')
    def test_get_astrometry(self, mock_run_query):
        mock_run_query.return_value = QTable({'col1': [1, 2], 'col2': [3, 4]})
        cluster = Cluster('ngc104')
        result = self.gaia_query.get_astrometry(0.1, gc=cluster)
        self.assertIsNotNone(result)
        mock_run_query.assert_called_once()

    @patch('grasp.gaia.query.GaiaQuery._run_query')
    def test_get_photometry(self, mock_run_query):
        mock_run_query.return_value = DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        cluster = Cluster('ngc104')
        result = self.gaia_query.get_photometry(0.1, gc=cluster)
        self.assertIsNotNone(result)
        mock_run_query.assert_called_once()

    @patch('grasp.gaia.query.GaiaQuery._run_query')
    def test_get_rv(self, mock_run_query):
        mock_run_query.return_value = QTable({'col1': [1, 2], 'col2': [3, 4]})
        cluster = Cluster('ngc104')
        result = self.gaia_query.get_rv(0.1, gc=cluster)
        self.assertIsNotNone(result)
        mock_run_query.assert_called_once()

if __name__ == '__main__':
    unittest.main()