#!/usr/bin/env python
# coding: utf-8

# In[4]:
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from src.finance_ml.indicators.indicators import Indicators

class ElasticNetModel:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None

    def load_data(self):
        print(self.data_file)
        
        #Loading the input Data.
        #This function accepts the file path and assigns data to dataframe.
        
        self.df = pd.read_parquet(self.data_file)

    def clean_data(self):
        
        #Preprocesing/Cleaning the Data.
        #Missing values and unnecessary columns are handled.
        
        self.df.drop('DATE', axis=1, inplace=True)
        self.df.fillna(0, inplace=True)

    def implement_indicators(self):
        
        #Implementing Indicators to the Data.
        #We have implemented VWAP and MFI Indicators in this model.
        
        ind = Indicators(ticker='', norm_data=True, calc_all=False, list_ind=["VWAP", "MFI"])
        self.df = ind.fit_transform(self.df)

    def split_data(self):
        
        #Spliting the Data to Train and Test Dataset.
        
        self.df.fillna(0, inplace=True)
        x = self.df.iloc[:, [col for col in range(len(self.df.columns)) if col != 3]]
        y = self.df.iloc[:, 3]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, random_state=42)

    def train_model(self):
        
        #Training the ElasticNet model using alpha and l1-ratio. 
        #A l1_ratio of 1 corresponds to pure Lasso regularization, while a l1_ratio of 0 corresponds to pure Ridge regularization. 
        #Intermediate values allow for Elastic net regularization.
        
        self.enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
        self.enet.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        
        #Evaluating the performance of the prediction made by ElasticNet model by using:
        #Mean Squared Error (MSE)
        #Mean Absolute Error (MAE)
        #Root Mean Squared Error (RMSE)

        y_pred = self.enet.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        return mse, mae, rmse

    def hyperparameter_tuning(self):
        
        #Evaluation for best estimation hyperparameters.
        #We have used Grid Search approach to search through different combinations of alpha and l1-ratio.
        #To find the best combination of alpha and l1-ratio which increases the model's predictive accuracy.

        el_net_grid = {'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
        elastic_cv = GridSearchCV(self.enet, el_net_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        elastic_cv.fit(self.x_train, self.y_train)
        y_pred2 = elastic_cv.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred2)
        mae = mean_absolute_error(self.y_test, y_pred2)
        rmse = np.sqrt(mse)
        best_estimator = elastic_cv.best_estimator_
        return mse, mae, rmse, best_estimator


# In[ ]:




