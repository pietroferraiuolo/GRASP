import unittest
import os
from unittest.mock import patch
from astropy.table import Table
from grasp.core import osutils as osu
from grasp.core import folder_paths as fn
import numpy as np

class TestOsutils(unittest.TestCase):

    @patch('astropy.table.QTable.read')
    @patch('grasp.core.osutils._findTracknum')
    def test_load_data(self, mock_findTracknum, mock_read):
        # Mock the _findTracknum function
        mock_findTracknum.return_value = '/mock/path'

        # Mock the QTable.read function
        mock_table = Table(names=('col1', 'col2'), rows=[(1, 2), (3, 4)])
        mock_read.return_value = mock_table

        # Test the load_data function
        result = osu.load_data(tn='dummy_tracking_number', as_sample=False)

        # Assertions
        self.assertTrue(np.array_equal(result['col1'], mock_table['col1']))
        self.assertTrue(np.array_equal(result['col2'], mock_table['col2']))
        #mock_findTracknum.assert_called_once_with('dummy_tracking_number', complete_path=True)
        mock_read.assert_called_once_with('/mock/path/query_data.fits', format='fits')

    def test_get_kwargs(self):
        # Test with key present in kwargs
        kwargs = {'key1': 'value1', 'key2': 'value2'}
        result = osu.get_kwargs(('key1', 'key3'), 'default', kwargs)
        self.assertEqual(result, 'value1')
        # Test with key not present in kwargs
        result = osu.get_kwargs(('key3', 'key4'), 'default', kwargs)
        self.assertEqual(result, 'default')

    @patch('os.listdir')
    @patch('os.path.join')
    @patch('grasp.core.osutils._findTracknum')
    def test_get_file_list(self, mock_findTracknum, mock_join, mock_listdir):
        # Mock the listdir and join functions
        mock_listdir.return_value = ['file1.txt', 'file2_data.txt', 'file3.txt']
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_findTracknum.return_value = '/basepath/folder1/tn1'

        # Test with tn and key
        result = osu.get_file_list(tn='tn1', key='_data')
        expected = '/basepath/folder1/tn1/file2_data.txt'
        self.assertEqual(result, expected)

        # Test with fold and no key
        result = osu.get_file_list(fold='/basepath/folder1')
        expected = ['/basepath/folder1/file1.txt', '/basepath/folder1/file2_data.txt', '/basepath/folder1/file3.txt']
        self.assertEqual(result, expected)

    @patch('os.listdir')
    @patch('grasp.core.osutils._fn.CLUSTER_DATA_FOLDER')
    def test_tnlist(self, mock_cluster_data_folder, mock_listdir):
        # Mock the CLUSTER_DATA_FOLDER and listdir functions
        mock_cluster_data_folder.return_value = '/basepath/gc_name'
        mock_listdir.return_value = ['tn1', 'tn2', 'tn3']

        # Test the tnlist function
        result = osu.tnlist('gc_name')
        expected = ['/basepath/gc_name/tn1', '/basepath/gc_name/tn2', '/basepath/gc_name/tn3']
        self.assertEqual(result, expected)

    def test_timestamp(self):
        # Test the _timestamp function
        result = osu.timestamp()
        # Check if the result matches the expected format
        self.assertRegex(result, r'\d{8}_\d{6}')

    @patch('os.listdir')
    @patch('os.path.join')
    def test_findTracknum(self, mock_join, mock_listdir):
        # Mock the listdir and join functions
        mock_join.side_effect = lambda basepath, folder: f"{basepath}/{folder}"
        def mock_listdir_res():
            mock_listdir.side_effect = [
                ['query'],
                ['gc_folder1', 'gc_folder2', 'gc_folder3'],  # First call to os.listdir
                ['tn_1', 'tn_2'],  # Second call to os.listdir for gc_folder1
                ['tn_2'],  # Second call to os.listdir for gc_folder2
                ['tn_3']  # Second call to os.listdir for gc_folder3
            ]
        # Test with complete_path=False
        mock_listdir_res()
        result = osu._findTracknum('tn_1', complete_path=False)
        expected = 'gc_folder1'
        self.assertEqual(result, expected)
        # Test with multiple folders
        mock_listdir_res()
        result2 = osu._findTracknum('tn_2', complete_path=False)
        expected2 = ['gc_folder1','gc_folder2']
        self.assertEqual(result2, expected2)
        # Test with complete_path=True
        mock_listdir_res()
        result = osu._findTracknum('tn_3', complete_path=True)
        expected = os.path.join(
            os.path.join(
                fn.QUERY_DATA_FOLDER,
                'gc_folder3'
            ), 
            'tn_3'
        )
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()