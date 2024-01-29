import sys
sys.path.append('../../../../Documents/')
import pytest
import pandas as pd
import numpy as np
from src.finance_ml.data_preparation.data_preparation import DataLoader

@pytest.fixture
def sample_data():
    """
    Fixture to provide sample data for testing.
    """
    # Load the data
    dataloader = DataLoader(time_index_col='DATE', keep_cols=['VOLUME', 'OPEN', 'CLOSE', 'LOW', 'TRANSACTIONS'])
    
    # Load datasets
    df = dataloader.load_dataset({'GOLD':'../data/commodities/GLD_2020-04-07_2022-04-06.parquet'})
    return df  

def test_data_loading(sample_data):
    """
    Test if the DataFrame is correctly created.
    """
    assert not sample_data.empty, "Dataframe is empty"
    expected_cols = ['GOLD_VOLUME', 'GOLD_OPEN', 'GOLD_CLOSE', 'GOLD_LOW', 'GOLD_TRANSACTIONS']

    for col in expected_cols:
        assert col in sample_data.columns, f"{col} column is missing"

def test_daily_returns_calculation(sample_data):
    """
    Test if daily returns are calculated correctly.
    """
    sample_data[f'Daily_Return_{"GOLD_CLOSE"}'] = sample_data["GOLD_CLOSE"].pct_change(fill_method=None).abs()    
    sample_data[f'Daily_Return_{"GOLD_CLOSE"}'] = sample_data[f'Daily_Return_{"GOLD_CLOSE"}'].fillna(0)

    assert sample_data[f'Daily_Return_{"GOLD_CLOSE"}'].isnull().sum() == 0, f'Null values found in Daily_Return_{"GOLD_CLOSE"}'

def test_weight_by_return_calculation(sample_data):
    """
    Test if weights by return are calculated correctly.
    """
   
    daily_return_col = f'Daily_Return_{"GOLD_CLOSE"}'
    weight_by_return_col = f'weight_by_return_{"GOLD_CLOSE"}'

    sample_data[daily_return_col] = sample_data["GOLD_CLOSE"].pct_change(fill_method=None).abs()
    sample_data[daily_return_col] = sample_data[daily_return_col].fillna(0)
    sample_data[weight_by_return_col] = sample_data[daily_return_col] / sample_data[daily_return_col].max()

    assert sample_data[weight_by_return_col].between(0, 1).all(), f'Weights are not in the range [0, 1] for {"GOLD_CLOSE"}'

def test_weight_by_time_decay_calculation(sample_data):
    """
    Test if weights by time decay are calculated correctly.
    """

    decay_rate = 0.01
    latest_date = sample_data.index.max()

    days_from_latest_col = f'Days_From_Latest_GOLD_CLOSE'
    weight_by_time_decay_col = f'weight_by_time_decay_GOLD_CLOSE'

    sample_data[days_from_latest_col] = (latest_date - sample_data.index).days
    sample_data[weight_by_time_decay_col] = np.exp(-decay_rate * sample_data[days_from_latest_col])

    assert sample_data[weight_by_time_decay_col].between(0, 1).all(), 'Weights are not in the range [0, 1] for GOLD_CLOSE'
    
    # Check if the average weight in the earlier part of the time series is generally less than in the later part
    mid_point = len(sample_data) // 2
    avg_early_weight = sample_data[weight_by_time_decay_col][:mid_point].mean()
    avg_late_weight = sample_data[weight_by_time_decay_col][mid_point:].mean()
    assert avg_early_weight < avg_late_weight, 'Weights do not generally increase over time for GOLD_CLOSE'
