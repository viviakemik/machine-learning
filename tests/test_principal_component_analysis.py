import unittest
import pandas as pd
import numpy as np
from src.finance_ml.principal_component_analysis.principal_component_analysis import PrincipalComponentAnalysis
import sys
sys.path.append('./')
from IPython.display import display, Math, Latex

class TestPCA(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame
        np.random.seed(0)
        data = {
            'AAPL_HIGH': np.random.randn(100),
            'AAPL_LOW': np.random.randn(100),
            'AAPL_VOLUME': np.random.randn(100)
        }
        self.df = pd.DataFrame(data)

    def test_pca_initialization(self):
        pca_instance = PrincipalComponentAnalysis(self.df)
        self.assertIsNotNone(pca_instance, "PCA instance should not be None")

    def test_pca_application(self):
        pca_instance = PrincipalComponentAnalysis(self.df)
        pca_instance.apply_pca(self.df)
        self.assertIsNotNone(pca_instance.pca_result, "PCA result should not be None")
        self.assertLessEqual(pca_instance.pca_result.shape[1], self.df.shape[1], "PCA result should have equal or less columns than original data")

    def test_variance_explained(self):
        pca_instance = PrincipalComponentAnalysis(self.df)
        pca_instance.apply_pca(self.df)
        total_variance_explained = sum(pca_instance.pca.explained_variance_ratio_)
        self.assertGreater(total_variance_explained, 0.0, "Total variance explained should be greater than 0")

if __name__ == '__main__':
    unittest.main()
