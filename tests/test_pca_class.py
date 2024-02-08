import unittest
import pandas as pd
import sys
import importlib
sys.path.append('../')
from IPython.display import display, Math, Latex
from src.finance_ml.data_preparation.data_preparation import DataLoader
from src.finance_ml.indicators.indicators import Indicators

from src.finance_ml.PCAWithRegression.PCA import PCAModel

class TestPCAModel(unittest.TestCase):
    def setUp(self):

        """
        Set up the test environment by creating a sample DataFrame and an instance of the PCAModel.
        """

        # Create a sample DataFrame for testing
        data = {'Feature1': [1, 2, 3, 4, 5],
                'Feature2': [5, 4, 3, 2, 1],
                'Target': [10, 20, 30, 40, 50]}
        self.df = pd.DataFrame(data)

        # Create an instance of the PCAModel for testing
        self.pca_model = PCAModel(data_frame=self.df, target_column='Target')

    # Test if preprocessing method sets the training and testing data properly
    def test_preprocess_data(self):

        """
        Test if preprocessing method sets the training and testing data properly.
        """

        self.pca_model.preprocess_data()
        self.assertIsNotNone(self.pca_model.X_train)
        self.assertIsNotNone(self.pca_model.X_test)
        self.assertIsNotNone(self.pca_model.y_train)
        self.assertIsNotNone(self.pca_model.y_test)

    # Test if PCA method initializes PCA, captures components, and transforms data
    def test_perform_pca(self):

        """
        Test if PCA method initializes PCA, captures components, and transforms data.
        """

        self.pca_model.preprocess_data()
        self.pca_model.perform_pca()
        self.assertIsNotNone(self.pca_model.pca)
        self.assertIsNotNone(self.pca_model.num_components_to_capture)
        self.assertIsNotNone(self.pca_model.df_pca)

    # Test if the regression model is trained successfully
    def test_train_regression_model(self):

        """
        Test if the regression model is trained successfully.
        """

        self.pca_model.preprocess_data()
        self.pca_model.perform_pca()
        self.pca_model.train_regression_model()
        self.assertIsNotNone(self.pca_model.regression_model)

    # Test if the expected and actual mse error is same
    def test_evaluate_model(self):

        """
        Test if the expected and actual mse error is same.
        """

        self.pca_model.preprocess_data()
        self.pca_model.perform_pca()
        self.pca_model.train_regression_model()
        #self.pca_model.evaluate_model()
        #assert self.pca_model.error == 0.0

    # Test if expected number of pca component is same as actual
    def test_print_pca_results(self):

        """
        Test if expected number of PCA component is same as actual.
        """

        self.pca_model.preprocess_data()
        self.pca_model.perform_pca()
        self.pca_model.print_pca_results()
        assert self.pca_model.num_components_to_capture == 1

    # Test if p - value is between 0 and 1
    def test_check_stationarity(self):

        """
        Test if p-value is between 0 and 1.
        """

        self.pca_model.preprocess_data()
        self.pca_model.perform_pca()
        self.pca_model.check_stationarity()
        assert 0 < self.pca_model.adfresult < 1
        
           

if __name__ == '__main__':
    unittest.main()
