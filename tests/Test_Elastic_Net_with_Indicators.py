#!/usr/bin/env python
# coding: utf-8

# In[5]:


#pip install pytest


# In[7]:


import pytest
import pandas as pd
from datetime import date
from sklearn.linear_model import ElasticNet
from src.finance_ml.Elastic_Net_with_Indicators.Elastic_Net_with_Indicators import ElasticNetModel

# Sample data for testing
data = {
    'VOLUME': [100, 200, 300, 400],
    'VW': [50, 60, 70, 80],
    'OPEN': [30, 40, 50, 60],
    'CLOSE': [35, 45, 55, 65],
    'HIGHT': [40, 50, 60, 70],
    'LOW': [25, 35, 45, 55],
    't': [1, 2, 3, 4],
    'TRANSACTIONS': [10, 20, 30, 40],
    'a': [1, 2, 3, 4],
    'op': [0, 1, 0, 1],
    'DATE': []
}

for d in ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']:
    data['DATE'].append(date.fromisoformat(d))

#@pytest.fixture
#def sample_data():
#    return pd.DataFrame(data)

@pytest.fixture()
def sample_data_file(tmp_path):
    df = pd.DataFrame(data)
    #df.set_index('HIGHT')
    file_path = tmp_path / 'sample_data.parquet'
    df.to_parquet(file_path)
    return file_path

def test_load_data(sample_data_file):
    model = ElasticNetModel(sample_data_file)
    model.load_data()
    #print(model.df.head, data.head)
    assert model.df.equals(pd.DataFrame(data))

def test_clean_data(sample_data_file):
    model = ElasticNetModel(sample_data_file)
    model.load_data()
    #model.df = sample_data
    model.clean_data()
    assert 'DATE' not in model.df.columns
    assert model.df.isnull().sum().sum() == 0

def test_implement_indicators(sample_data_file):
    model = ElasticNetModel(sample_data_file)
    model.load_data()
    #model.df = sample_data
    model.clean_data()
    model.implement_indicators()
    print(model.df.columns)
    assert 'VWAP_w14' in model.df.columns
    assert 'MFI_w14' in model.df.columns

# Add more tests for other functions similarly...

def test_split_data(sample_data_file):
    model = ElasticNetModel(sample_data_file)
    model.load_data()
    model.clean_data()
    #model.df = sample_data
    model.implement_indicators()
    model.split_data()
    assert model.x_train.shape[0] == model.y_train.shape[0]
    assert model.x_test.shape[0] == model.y_test.shape[0]

def test_train_model(sample_data_file):
    model = ElasticNetModel(sample_data_file)
    model.load_data()
    model.clean_data()
    #model.df = sample_data
    model.implement_indicators()
    model.split_data()
    model.train_model()
    assert hasattr(model, 'enet')

def test_evaluate_model(sample_data_file):
    model = ElasticNetModel(sample_data_file)
    model.load_data()
    model.clean_data()
    #model.df = sample_data
    model.implement_indicators()
    model.split_data()
    model.train_model()
    mse, mae, rmse = model.evaluate_model()
    assert isinstance(mse, float)
    assert isinstance(mae, float)
    assert isinstance(rmse, float)

def test_hyperparameter_tuning(sample_data_file):
    model = ElasticNetModel(sample_data_file)
    model.load_data()
    model.clean_data()
    #model.df = sample_data
    model.implement_indicators()
    model.split_data()
    model.train_model()
    mse, mae, rmse, best_estimator = model.hyperparameter_tuning()
    assert isinstance(mse, float)
    assert isinstance(mae, float)
    assert isinstance(rmse, float)
    assert isinstance(best_estimator, ElasticNet)

if __name__ == "__main__":
    pytest.main()


# In[ ]:




