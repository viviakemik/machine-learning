import unittest
from optimal_number_of_clusters import load_data, preprocess_data, kmeans_silhouette_scores, kmeans_calinski_scores
from optimal_number_of_clusters import calculate_gap_statistic, \
    gap_statistic_visualization
import pandas as pd
import numpy as np

class TestFinanceML(unittest.TestCase):

    def setUp(self):
        self.file_path = 'dataset.parquet'
        self.real_estate_data = load_data(self.file_path)

        # Take the first 2000 entries from the dataset
        self.real_estate_data = self.real_estate_data.head(2000)
        self.df_real_estate = pd.DataFrame({'clusters': [1, 2, 2, 1, 2, 1, 1, 2, 2, 1]})
        self.df_real_estate = preprocess_data(self.real_estate_data)

    def test_load_data(self):
        self.assertIsNotNone(self.real_estate_data)
        self.assertIsInstance(self.real_estate_data, pd.DataFrame)

    def test_preprocess_data(self):
        processed_data = preprocess_data(self.real_estate_data)
        self.assertIsNotNone(processed_data)
        # Add more specific assertions based on your preprocessing logic

    def test_kmeans_silhouette_scores(self):
        k_values = range(2, 10)
        silhouette_scores = kmeans_silhouette_scores(self.df_real_estate, k_values)
        self.assertEqual(len(silhouette_scores), len(k_values))

    def test_kmeans_calinski_scores(self):
        k_values = range(2, 10)
        calinski_scores = kmeans_calinski_scores(self.df_real_estate, k_values)
        self.assertEqual(len(calinski_scores), len(k_values))



    def test_gap_statistic_visualization(self):
        k_values = range(2, 10)
        with self.assertRaises(Exception):
            gap_statistic_visualization(self.df_real_estate, k_values)

    def test_calculate_gap_statistic(self):
        k = 3  # Provide a specific k value for testing
        labels = np.random.randint(0, k, len(self.df_real_estate))
        gap_statistic = calculate_gap_statistic(self.df_real_estate, labels, k)
        self.assertIsInstance(gap_statistic, float)

if __name__ == '__main__':
    unittest.main()
