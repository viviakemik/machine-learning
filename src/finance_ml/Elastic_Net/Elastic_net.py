import sys
import importlib
sys.path.append('../')

from IPython.display import display, Math, Latex

from src.finance_ml.data_preparation.data_preparation import DataLoader
from src.finance_ml.indicators.indicators import Indicators

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

'''
Contributors:
1. Nilay Khare 
2. Vipul Tank 
3. Anmol Tiwari
'''

class RegressionAnalysis:
    def __init__(self, data_file, ticker, time_index_col='DATE', keep_cols=None, degree=1, test_size=0.3, random_state=101):
        """
        Initialize the RegressionAnalysis object.

        Parameters:
        - data_file: str, file path to the financial data
        - ticker: str, the stock ticker symbol
        - time_index_col: str, the column representing the time index in the dataset
        - keep_cols: list, columns to keep in the dataset
        - degree: int, degree of polynomial features to generate
        - test_size: float, the proportion of the dataset to include in the test split
        - random_state: int, seed for random number generation

        Returns:
        None
        """
        self.data_file = data_file
        self.ticker = ticker
        self.time_index_col = time_index_col
        self.keep_cols = keep_cols
        self.degree = degree
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.poly_features = None
        self.scaler = None

    def load_data(self):
        """
        Load financial data from the specified file and prepare the dataset.

        Returns:
        None
        """
        dataloader = DataLoader(time_index_col=self.time_index_col, keep_cols=self.keep_cols)
        self.df = dataloader.load_dataset({self.ticker: '../data/' + self.data_file})

    def prepare_features(self):
        """
        Prepare polynomial features and scale the data.

        Returns:
        None
        """
        X = self.df.drop([f'{self.ticker}_CLOSE'], axis=1)
        y = self.df[f'{self.ticker}_CLOSE']
        polynomial_converter = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.poly_features = polynomial_converter.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.poly_features, y, test_size=self.test_size, random_state=self.random_state)
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_ridge(self, alpha=10):
        """
        Train a Ridge Regression model.

        Parameters:
        - alpha: float, regularization strength

        Returns:
        Ridge model
        """
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(self.X_train, self.y_train)
        return ridge_model

    def train_ridge_cv(self, alphas=(0.1, 1.0, 10.0)):
        """
        Train a Ridge Regression model with cross-validation.

        Parameters:
        - alphas: tuple, regularization strengths

        Returns:
        RidgeCV model
        """
        ridge_cv_model = RidgeCV(alphas=alphas, scoring='neg_mean_absolute_error')
        ridge_cv_model.fit(self.X_train, self.y_train)
        return ridge_cv_model

    def train_lasso_cv(self, eps=0.01, n_alphas=100, cv=5):
        """
        Train a Lasso Regression model with cross-validation.

        Parameters:
        - eps: float, path length
        - n_alphas: int, number of alphas along the regularization path
        - cv: int, number of cross-validation folds

        Returns:
        LassoCV model
        """
        lasso_cv_model = LassoCV(eps=eps, n_alphas=n_alphas, cv=cv)
        lasso_cv_model.fit(self.X_train, self.y_train)
        return lasso_cv_model

    def train_elastic_net_cv(self, l1_ratios=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1), max_iter=100000, cv=5):
        """
        Train an Elastic Net Regression model with cross-validation.

        Parameters:
        - l1_ratios: tuple, mixing parameters
        - max_iter: int, maximum number of iterations
        - cv: int, number of cross-validation folds

        Returns:
        ElasticNetCV model
        """
        elastic_model = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, max_iter=max_iter)
        elastic_model.fit(self.X_train, self.y_train)
        return elastic_model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the performance of a regression model.

        Parameters:
        - model: trained regression model
        - X_test: test features
        - y_test: true labels for the test set

        Returns:
        DataFrame with evaluation metrics
        """
        y_pred = model.predict(X_test)
        MAE = mean_absolute_error(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(MSE)
        return pd.DataFrame([MAE, MSE, RMSE], index=['MAE', 'MSE', 'RMSE'], columns=['metrics'])

    def count_nonzero_coefficients(self, model):
        """
        Count the number of non-zero coefficients in a trained regression model.

        Parameters:
        - model: trained regression model

        Returns:
        int, number of non-zero coefficients
        """
        return np.count_nonzero(model.coef_)

    def display_results(self, model, model_name):
        """
        Display the results of a trained regression model.

        Parameters:
        - model: trained regression model
        - model_name: str, name of the model

        Returns:
        None
        """
        y_pred = model.predict(self.X_test)
        metrics_df = self.evaluate_model(model, self.X_test, self.y_test)
        coef_count = self.count_nonzero_coefficients(model)
        print(f"After {model_name}, we have only {coef_count} non-zero coefficients.")
        display(metrics_df)
