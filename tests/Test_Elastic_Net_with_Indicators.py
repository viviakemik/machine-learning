#!/usr/bin/env python
# coding: utf-8

# In[5]:


#pip install pytest


# In[7]:


import pytest
import pandas as pd
from Elastic_Net_with_Indicators import ElasticNetModel

# Sample data for testing
data = {
    'VOLUME': [100, 200, 300, 400],
    'VW': [50, 60, 70, 80],
    'OPEN': [30, 40, 50, 60],
    'CLOSE': [35, 45, 55, 65],
    'HIGH': [40, 50, 60, 70],
    'LOW': [25, 35, 45, 55],
    't': [1, 2, 3, 4],
    'TRANSACTIONS': [10, 20, 30, 40],
    'a': [1, 2, 3, 4],
    'op': [0, 1, 0, 1],
    'DATE': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']
}

@pytest.fixture
def sample_data():
    return pd.DataFrame(data)

def test_load_data(sample_data):
    model = ElasticNetModel('sample_data.parquet')
    model.load_data()
    assert model.df.equals(sample_data)

def test_clean_data(sample_data):
    model = ElasticNetModel('sample_data.parquet')
    model.df = sample_data
    model.clean_data()
    assert 'DATE' not in model.df.columns
    assert model.df.isnull().sum().sum() == 0

def test_implement_indicators(sample_data):
    model = ElasticNetModel('sample_data.parquet')
    model.df = sample_data
    model.implement_indicators()
    assert 'VWAP' in model.df.columns
    assert 'MFI' in model.df.columns

# Add more tests for other functions similarly...

if __name__ == "__main__":
    pytest.main()


# In[ ]:




