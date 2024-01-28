import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class PCAModel:
    def __init__(self, data_frame, target_column, variance_ratio=0.95, test_size=0.2, random_state=42):
        self.data_frame = data_frame
        self.target_column = target_column
        self.variance_ratio = variance_ratio
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pca = None
        self.num_components_to_capture = None
        self.df_pca = None
        self.regression_model = None
        self.adfresult = None
        self.error = None

    def preprocess_data(self):
        df_pca = self.data_frame.dropna(axis=0)
        X = df_pca.drop(self.target_column, axis=1)
        y = df_pca[self.target_column]
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def perform_pca(self):
        # Initialize PCA with the number of components set to None
        self.pca = PCA(n_components=None)

        # Fit the PCA model on your data
        self.pca.fit(self.X_train)

        # Calculate the cumulative sum of explained variance ratios
        cumulative_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)

        # Determine the number of components needed to capture the desired variance
        self.num_components_to_capture = np.argmax(cumulative_variance_ratio >= self.variance_ratio) + 1

        # Use the determined number of components to perform PCA on training and testing sets
        self.pca = PCA(n_components=self.num_components_to_capture)
        self.X_train = self.pca.fit_transform(self.X_train)
        self.X_test = self.pca.transform(self.X_test)

        # Convert the transformed data into a DataFrame with column names
        column_names_after_pca = [f"PC{i}" for i in range(1, self.num_components_to_capture + 1)]
        self.df_pca = pd.DataFrame(data=self.X_train, columns=column_names_after_pca)

    def train_regression_model(self):
        # Train a linear regression model
        self.regression_model = LinearRegression()
        self.regression_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Make predictions on the test set
        y_pred = self.regression_model.predict(self.X_test)

        # Evaluate the performance of the regression model
        mse = mean_squared_error(self.y_test, y_pred)
        self.error = mse
        print(f'Mean Squared Error: {mse}')

        num_predictors = self.X_train.shape[1]

        # fig, axs = plt.subplots(nrows=num_predictors, ncols=1, figsize=(5, 5 * num_predictors), sharey=True)

        # for i in range(num_predictors):
        #     axs[i].scatter(self.X_test[:, i], self.y_test, color='blue', label='Actual Data')
        #     axs[i].scatter(self.X_test[:, i], y_pred, color='red', label='Predicted Data')
        #     axs[i].set_xlabel(f'PC-{i+1}')
        #     axs[i].set_ylabel('USDBRL_CLOSE_returns_norm')
        #     axs[i].legend()

        # plt.tight_layout()
        # plt.show()

    def print_pca_results(self):
        # Print the results of PCA
        print(f"Number of components to capture {self.variance_ratio * 100}% variance: {self.num_components_to_capture}")
        print(f"Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")

        for i in range(self.num_components_to_capture):
            print(f"Explained Variance Ratio for Principal Component {i + 1}: {np.sum(self.pca.explained_variance_ratio_[:i+1]):.4f}")


    def check_stationarity(self):
        for column_name in self.df_pca.columns:
            # Check stationarity using Augmented Dickey-Fuller test
            result = adfuller(self.df_pca[column_name])

            # Print ADF test results for each column
            print(f'ADF Statistic for {column_name}: {result[0]}')
            print(f'p-value for {column_name}: {result[1]}')
            print(f'Critical Values for {column_name}: {result[4]}')

            # Interpret the results
            if result[1] <= 0.05:
                print(f'The p-value ({result[1]:.4f}) is less than or equal to 0.05. Reject the null hypothesis.')
                print(f'{column_name} is likely stationary.\n')
            else:
                print(f'The p-value ({result[1]:.4f}) is greater than 0.05. Fail to reject the null hypothesis.')
                print(f'{column_name} may not be stationary.\n')

        self.adfresult = result[1]
