import unittest
import pandas as pd
import numpy as np

from src.finance_ml.Elastic_Net.Elastic_net import RegressionAnalysis
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, ElasticNetCV
from sklearn.datasets import make_regression
import unittest

'''
Contributors:
1. Nilay Khare 
2. Vipul Tank 
3. Anmol Tiwari
'''

class TestRegressionAnalysis(unittest.TestCase):
    """
    Test class for the RegressionAnalysis class.

    This class defines test cases for the various regression models in the RegressionAnalysis class.
    """

    def setUp(self):
        """
        Set up the necessary parameters and create an instance of the RegressionAnalysis class.

        This method is executed before each test case.
        """
        # Modify the file paths and parameters accordingly
        self.fname_USDBRL = 'FX/USDBRL_2020-04-07_2022-04-06.parquet'
        self.N = 40000
        self.ticker = 'USDBRL'
        self.keep_cols = ['VOLUME', 'VW', 'OPEN', 'CLOSE', 'HIGHT', 'LOW', 't', 'TRANSACTIONS']

        self.regression_analysis = RegressionAnalysis(data_file=self.fname_USDBRL, ticker=self.ticker, keep_cols=self.keep_cols)
        self.regression_analysis.load_data()
        self.regression_analysis.prepare_features()

    def test_ridge_regression(self):
        """
        Test the Ridge regression model.

        This method tests the training of the Ridge regression model and evaluates its performance metrics.
        """
        ridge_model = self.regression_analysis.train_ridge(alpha=10)
        metrics_df = self.regression_analysis.evaluate_model(ridge_model, self.regression_analysis.X_test, self.regression_analysis.y_test)
        coef_count = self.regression_analysis.count_nonzero_coefficients(ridge_model)
        self.assertTrue(coef_count > 0)
        self.assertEqual(metrics_df.shape, (3, 1))

    def test_ridge_cv_regression(self):
        """
        Test the RidgeCV regression model.

        This method tests the training of the RidgeCV regression model and evaluates its performance metrics.
        """
        ridge_cv_model = self.regression_analysis.train_ridge_cv(alphas=(0.1, 1.0, 10.0))
        metrics_df = self.regression_analysis.evaluate_model(ridge_cv_model, self.regression_analysis.X_test, self.regression_analysis.y_test)
        coef_count = self.regression_analysis.count_nonzero_coefficients(ridge_cv_model)
        self.assertTrue(coef_count > 0)
        self.assertEqual(metrics_df.shape, (3, 1))

    def test_lasso_cv_regression(self):
        """
        Test the LassoCV regression model.

        This method tests the training of the LassoCV regression model and evaluates its performance metrics.
        """
        lasso_cv_model = self.regression_analysis.train_lasso_cv(eps=0.01, n_alphas=100, cv=5)
        metrics_df = self.regression_analysis.evaluate_model(lasso_cv_model, self.regression_analysis.X_test, self.regression_analysis.y_test)
        coef_count = self.regression_analysis.count_nonzero_coefficients(lasso_cv_model)
        self.assertTrue(coef_count > 0)
        self.assertEqual(metrics_df.shape, (3, 1))

    def test_elastic_net_regression(self):
        """
        Test the ElasticNetCV regression model.

        This method tests the training of the ElasticNetCV regression model and evaluates its performance metrics.
        """
        elastic_model = self.regression_analysis.train_elastic_net_cv(l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], max_iter=100000)
        metrics_df = self.regression_analysis.evaluate_model(elastic_model, self.regression_analysis.X_test, self.regression_analysis.y_test)
        coef_count = self.regression_analysis.count_nonzero_coefficients(elastic_model)
        self.assertTrue(coef_count > 0)
        self.assertEqual(metrics_df.shape, (3, 1))
        
#test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRegressionAnalysis)
#unittest.TextTestRunner().run(test_suite)

test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRegressionAnalysis)
unittest.TextTestRunner().run(test_suite)