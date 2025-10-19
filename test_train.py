import unittest
import pandas as pd
from src.train import train_and_evaluate
import os

class TestTrainModule(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset for testing
        self.data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': ['A', 'B', 'A', 'B', 'A'],
            'Drug': ['DrugA', 'DrugB', 'DrugA', 'DrugB', 'DrugA']
        })
        self.data.to_csv('data/dataset.csv', index=False)

    def test_data_loading(self):
        data = pd.read_csv('data/dataset.csv')
        self.assertEqual(data.shape[0], 5)
        self.assertIn('Drug', data.columns)

    def test_train_and_evaluate(self):
        # This will run the training and evaluation process
        try:
            train_and_evaluate()
        except Exception as e:
            self.fail(f"train_and_evaluate raised an exception: {e}")

    def tearDown(self):
        os.remove('data/dataset.csv')

if __name__ == '__main__':
    unittest.main()