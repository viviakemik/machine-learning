import numpy as np
import pandas as pd
from unittest.mock import patch
import matplotlib.pyplot as plt
import sys
sys.path.append('../ML-in-Finance/src')
from finance_ml.frac_diff.fractional_differentiation import FinancialAnalysis 

def test_financial_analysis():
    # Initialize FinancialAnalysis with mock data
    financial_analysis = FinancialAnalysis('../data/cryptos/BTCUSD_2020-04-07_2022-04-06.parquet')
        
    # Test frac_diff_ffd function
    result = financial_analysis.frac_diff_ffd(col_name='OPEN', d=0.4, thres=1e-5)
    print("Result shape:", result.shape)
    print("DataFrame shape:", financial_analysis.df.shape)
    # assert result.shape == (len(financial_analysis.df) - 4,)  # Adjust the shape according to your expectation

# Run the test
if __name__ == '__main__':
    test_financial_analysis()