# import warnings
import pytest
import pandas as pd
import numpy as np
from src.finance_ml.clustering.Hierarchical import Hierarchical
from pandas import Timestamp


def test_hierarchical_init():
    '''
    Tests initialization of the Hierarchical class with a dataset, labels, and an interval.
    '''
    dataset = pd.DataFrame.from_dict(preprocessed_dataset_as_dict)
    labels = dataset.columns.tolist()
    interval = 24

    # Initialize the Hierarchical class
    hierarchical_model = Hierarchical(dataset, labels, interval)

    assert isinstance(hierarchical_model, Hierarchical), "The object should be an instance of Hierarchical"
    assert hierarchical_model.dataset.equals(dataset), "Dataset in Hierarchical should match the input dataset"
    assert hierarchical_model.labels == labels, "Labels should match the input labels"
    assert hierarchical_model.interval == interval, "Interval should match the input interval"


def test_preprocessing():
    '''
    Tests the preprocessing method of the Hierarchical class.
    '''
    dataset = pd.DataFrame.from_dict(na_test_dataset_as_dict)
    dataset.index = pd.to_datetime(dataset.index)
    labels = dataset.columns.tolist()

    hierarchical_model = Hierarchical(dataset, labels, interval=2)
    hierarchical_model.preprocessing(fillna_method='', resample='s')

    assert hierarchical_model.dataset.isna().sum().sum() == 0, "There should be no NaN values after preprocessing"
    assert not hierarchical_model.dataset.duplicated().any(), "There should be no duplicate values after preprocessing"


def prepare_na_dataset():
    '''
    Prepare the dataset of na type for testing.
    '''

    dataset = pd.DataFrame.from_dict(na_test_dataset_as_dict)
    labels = dataset.columns.tolist()
    dataset.index = pd.to_datetime(dataset.index)
    dataset.drop_duplicates(inplace=True)
    dataset = dataset.resample('T').interpolate()
    dataset.dropna(inplace=True)
    return dataset, labels


def test_apply_sliding_window():
    '''
    Tests the sliding window application on a dataset.
    '''
    dataset, labels = prepare_na_dataset()
    hierarchical_model = Hierarchical(None, labels)
    interval = 2
    hierarchical_model.apply_sliding_window(dataset, interval=interval)

    expected_length = len(dataset) - interval + 1
    assert len(
        hierarchical_model.returns_df) == expected_length, "Length of returns_df should match the expected length"


def test_standardize_dataset():
    '''
    Tests the standardization of a dataset.
    '''
    dataset, labels = prepare_na_dataset()
    hierarchical_model = Hierarchical(None, labels)
    standardized_df = hierarchical_model.standardize_dataset(dataset)

    assert standardized_df['GLD_OPEN'].min() >= 0, "Minimum value should be at least 0 after (min-max) standardization"
    assert standardized_df['GLD_OPEN'].max() <= 1, "Maximum value should be at most 1 after standardization"


def test_find_linkage():
    '''
    Tests the linkage finding in a dataset.
    '''
    dataset = pd.DataFrame.from_dict(preprocessed_dataset_as_dict)
    labels = dataset.columns.tolist()
    hierarchical_model = Hierarchical(None, labels)
    hierarchical_model.find_linkage(dataset)
    number_of_assets = dataset.shape[1]

    assert hierarchical_model.data_linkage.shape[
               0] == number_of_assets - 1, "Data linkage rows should be one less than the number of assets"


# short datasets to run the test cases
preprocessed_dataset_as_dict = {
    'GLD_OPEN': {0: 0.38464004122964823,
                 1: 0.3910964199996524,
                 2: 0.4087510561804713,
                 3: 0.3945266875774748,
                 4: 0.39244635959467405},
    'PDBC_OPEN': {0: 0.8214957954861211,
                  1: 0.8130919956695393,
                  2: 0.8091855364495322,
                  3: 0.8092291073132253,
                  4: 0.8092726453723661},
    'SLV_OPEN': {0: 0.025234306432803555,
                 1: 0.1013410109551156,
                 2: 0.13465383215147542,
                 3: 0.0,
                 4: 0.07328013739570804},
    'BTCUSD_OPEN': {0: 0.5731076205435772,
                    1: 0.5958208158044516,
                    2: 0.629059325456764,
                    3: 0.6432256324284117,
                    4: 0.6417925877208459},
    'DOGEUSD_OPEN': {0: 0.17032561572412075,
                     1: 0.15707951181127197,
                     2: 0.16124574612654935,
                     3: 0.1600567541016278,
                     4: 0.17109548795074772},
    'ETHUSD_OPEN': {0: 0.6330704912035702,
                    1: 0.6406197466670511,
                    2: 0.6673777852148696,
                    3: 0.6695780099935933,
                    4: 0.6429704119713471},
    'AAPL_OPEN': {0: 0.5466992685151293,
                  1: 0.575921932307875,
                  2: 0.5747366633621463,
                  3: 0.5906054530141398,
                  4: 0.587511497321409},
    'FB_OPEN': {0: 0.8100617889349626,
                1: 0.8139014305138565,
                2: 0.8205133043683487,
                3: 0.8134230536551971,
                4: 0.8122518851252357},
}

# dataset with na values to test certain scenarios
na_test_dataset_as_dict = {
    "GLD_OPEN": {
        "2020-04-07 12:17:00": 0.6642654665,
        "2020-04-07 12:18:00": 0.6376435328,
        "2020-04-07 12:19:00": np.nan,
        "2020-04-07 12:20:00": np.nan,
        "2020-04-07 12:21:00": np.nan,
        "2020-04-07 12:22:00": np.nan,
        "2020-04-07 12:23:00": np.nan,
        "2020-04-07 12:24:00": 0.197066755,
        "2020-04-07 12:25:00": np.nan,
        "2020-04-07 12:26:00": np.nan,
        "2020-04-07 12:27:00": np.nan,
        "2020-04-07 12:28:00": 0.9009368614,
        "2020-04-07 12:29:00": 0.3877997903,
        "2020-04-07 12:30:00": 0.7079662339,
        "2020-04-07 12:31:00": 0.2767255607,
        "2020-04-07 12:32:00": np.nan,
        "2020-04-07 12:33:00": np.nan,
        "2020-04-07 12:34:00": np.nan,
        "2020-04-07 12:35:00": np.nan,
        "2020-04-07 12:36:00": 0.0315305291,
        "2020-04-07 12:37:00": 0.1585705127,
        "2020-04-07 12:38:00": 0.3636897778,
        "2020-04-07 12:39:00": 0.3480869543,
        "2020-04-07 12:40:00": np.nan,
        "2020-04-07 12:41:00": np.nan,
        "2020-04-07 12:42:00": np.nan,
        "2020-04-07 12:43:00": 0.569797276,
        "2020-04-07 12:44:00": 0.1848692028,
        "2020-04-07 12:45:00": 0.3974779214,
        "2020-04-07 12:46:00": 0.9427790907
    }
}
