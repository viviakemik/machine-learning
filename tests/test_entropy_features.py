import unittest
import pandas as pd
from src.finance_ml.entropy_features.entropy_features import EntropyFeatures

class TestEntropyFeatures(unittest.TestCase):
    def setUp(self):
        # Set up test data (you may modify this according to your actual data structure)
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })

    def test_plugin_entropy_estimator(self):
        entropy_calculator = EntropyFeatures()

        # Test if an exception is raised when data is not loaded
        with self.assertRaises(ValueError):
            entropy_calculator.plugin_entropy_estimator(None)

        # Test the plugin_entropy_estimator method
        result = entropy_calculator.plugin_entropy_estimator(self.data)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_lempel_ziv_estimator(self):
        entropy_calculator = EntropyFeatures()

        # Test if an exception is raised when data is not loaded
        with self.assertRaises(ValueError):
            entropy_calculator.lempel_ziv_estimator(None)

        # Test the lempel_ziv_estimator method
        result = entropy_calculator.lempel_ziv_estimator(self.data)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()
