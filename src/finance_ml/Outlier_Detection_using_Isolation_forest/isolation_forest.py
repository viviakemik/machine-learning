# isolation_forest.py

import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path, dataloader, ticker, N):
    df = dataloader.load_dataset({ticker: file_path}).iloc[:N]
    return df

class OutlierDetector: # Setting contamination Value
    def __init__(self, contamination):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination)

    def fit_predict(self, data, column):# trained our model here
        data_copy = data.copy()
        data_copy['outliers_isolation_forest'] = self.isolation_forest.fit_predict(data[[column]])
        return data_copy

    def visualize_outliers(self, data, column): #viualize our outlier and inliers from Data
        plt.figure(figsize=(8, 6))
        plt.scatter(data.index[data['outliers_isolation_forest'] == 1], data[column][data['outliers_isolation_forest'] == 1],
                    c='green', label='Inliers')
        plt.scatter(data.index[data['outliers_isolation_forest'] == -1], data[column][data['outliers_isolation_forest'] == -1],
                    c='red', label='Outliers')
        plt.title(f'Isolation Forest Outliers Detection for {column}')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.legend()
        plt.show()

    def remove_outliers(self, data): # Remove outliers from the Data
        data_inliers = data[data['outliers_isolation_forest'] == 1]
        return data_inliers
