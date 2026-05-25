import unittest
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pandas as pd

from grasp._utility.cluster import Cluster


class TestCluster(unittest.TestCase):

    @patch('grasp.core.folder_paths.CLUSTER_DATA_FOLDER')
    @patch('grasp.core.folder_paths.CLUSTER_MODEL_FOLDER')
    @patch('grasp._utility.cluster.Cluster._load_cluster_parameters')
    @patch('grasp._utility.cluster.Cluster._load_king_model')
    def test_cluster_initialization(self, mock_load_king_model, mock_load_cluster_parameters, mock_model_folder, mock_data_folder):
        mock_load_cluster_parameters.return_value = pd.Series({
            'ra': 10.0,
            'dec': 20.0,
            'dist': 5.0,
            'rc': 0.5,
            'rh': 1.0,
            'w0': 8.82,
            'logc': 1.5,
            'collapsed': 'N'
        })
        mock_load_king_model.return_value = MagicMock()

        cluster = Cluster('ngc104')

        self.assertEqual(cluster.id, 'NGC104')
        self.assertEqual(cluster.ra, 10.0 * u.deg)
        self.assertEqual(cluster.dec, 20.0 * u.deg)
        self.assertEqual(cluster.dist, 5.0 * u.kpc)
        self.assertEqual(cluster.rc, 0.5 / 60 * u.deg)
        self.assertEqual(cluster.rh, 1.0 / 60 * u.deg)
        self.assertEqual(cluster.w0, 8.82)
        self.assertEqual(cluster.logc, 1.5)
        self.assertEqual(cluster.rt, (0.5 / 60 * u.deg) * 10**1.5)
        self.assertFalse(cluster.cflag)

    @patch('grasp.core.folder_paths.CLUSTER_DATA_FOLDER')
    @patch('grasp.core.folder_paths.CLUSTER_MODEL_FOLDER')
    @patch('grasp._utility.cluster.Cluster._load_cluster_parameters')
    @patch('grasp._utility.cluster.Cluster._load_king_model')
    def test_cluster_initialization_no_name(self, mock_load_king_model, mock_load_cluster_parameters, mock_model_folder, mock_data_folder):
        cluster = Cluster()

        self.assertEqual(cluster.id, 'UntrackedData')
        self.assertIsNone(cluster.ra)
        self.assertIsNone(cluster.dec)
        self.assertIsNone(cluster.model)

    @patch('grasp.core.folder_paths.CLUSTER_DATA_FOLDER')
    @patch('grasp.core.folder_paths.CLUSTER_MODEL_FOLDER')
    @patch('grasp._utility.cluster.Cluster._load_cluster_parameters')
    @patch('grasp._utility.cluster.Cluster._load_king_model')
    def test_show_model(self, mock_load_king_model, mock_load_cluster_parameters, mock_model_folder, mock_data_folder):
        mock_load_cluster_parameters.return_value = pd.Series({
            'ra': 10.0,
            'dec': 20.0,
            'dist': 5.0,
            'rc': 0.5,
            'rh': 1.0,
            'w0': 8.82,
            'logc': 1.5,
            'collapsed': 'N'
        })
        mock_load_king_model.return_value = {
            'xi': np.array([0, 0.5, 1]),
            'w': np.array([0, 0.5, 1])
        }

        cluster = Cluster('ngc104')
        cluster.show_model()

    @patch('grasp.core.folder_paths.CATALOG_FILE', 'mock_catalog.xlsx')
    @patch('pandas.read_excel')
    def test_load_cluster_parameters(self, mock_read_excel):
        mock_read_excel.return_value = pd.DataFrame({
            'ra': [10.0],
            'dec': [20.0],
            'dist': [5.0],
            'rc': [0.5],
            'rh': [1.0],
            'w0': [8.82],
            'logc': [1.5],
            'collapsed': ['N']
        }, index=['NGC104'])

        cluster = Cluster('ngc104')
        params = cluster._load_cluster_parameters('ngc104')

        self.assertEqual(params['ra'], 10.0)
        self.assertEqual(params['dec'], 20.0)
        self.assertEqual(params['dist'], 5.0)
        self.assertEqual(params['rc'], 0.5)
        self.assertEqual(params['rh'], 1.0)
        self.assertEqual(params['w0'], 8.82)
        self.assertEqual(params['logc'], 1.5)
        self.assertEqual(params['collapsed'], 'N')

if __name__ == '__main__':
    unittest.main()
