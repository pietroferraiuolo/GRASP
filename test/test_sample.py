import unittest
import numpy as np
from astropy.table import QTable
from grasp._utility.sample import Sample

class TestSample(unittest.TestCase):
    def setUp(self):
        data = QTable()
        data['a'] = np.arange(10)
        data['b'] = np.linspace(0, 1, 10)
        data['c'] = np.random.choice([np.nan, 1], size=10)
        self.sample = Sample(data, gc='UntrackedData')

    def test_drop_columns(self):
        self.sample.drop_columns(['b'])
        self.assertNotIn('b', self.sample.colnames)

    def test_apply_conditions_str(self):
        filtered = self.sample.apply_conditions("a > 5")
        self.assertTrue(np.all(filtered['a'] > 5))

    def test_apply_conditions_list(self):
        filtered = self.sample.apply_conditions(["a > 2", "b < 0.8"])
        self.assertTrue(np.all((filtered['a'] > 2) & (filtered['b'] < 0.8)))

    def test_apply_conditions_dict(self):
        filtered = self.sample.apply_conditions({'a': '> 2', 'b': '< 0.8'})
        self.assertTrue(np.all((filtered['a'] > 2) & (filtered['b'] < 0.8)))

    def test_apply_conditions_inplace(self):
        self.sample.apply_conditions({'a': '> 2', 'b': '< 0.8'}, inplace=True)
        self.assertTrue(np.all(self.sample['a'] > 2 ))

    def test_to_pandas_and_numpy(self):
        df = self.sample.to_pandas()
        arr = self.sample.to_numpy()
        self.assertEqual(df.shape[0], arr.shape[0])
        self.assertEqual(df.shape[1], arr.shape[1])
        self.assertEqual(df.shape[0], len(self.sample))
        self.assertEqual(df.shape[1], len(self.sample.colnames))

    def test_join(self):
        other = self.sample.copy()
        joined = self.sample.join(other)
        self.assertIsInstance(joined, Sample)
        for col in self.sample.colnames:
            self.assertIn(col, joined.colnames)

    # To review
    # def test_dropna(self):
    #     self.sample.dropna(inplace=True)
    #     self.assertFalse(np.isnan(self.sample['b']).any())

    def test_backup_and_reset(self):
        self.sample['a'] = self.sample['a'] + 100
        self.sample.reset_sample()
        np.testing.assert_array_equal(self.sample['a'], np.arange(10))

if __name__ == '__main__':
    unittest.main()
