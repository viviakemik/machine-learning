# """
# @Group: 
#     Yash Bhesaniya
#     Mit Desai
#     Bhumi Patel

# """


import numpy as np
import pandas as pd

class FinancialAnalysis:
    def __init__(self, data_path):
        """
        Initialize FinancialAnalysis object with the provided data path.

        Parameters:
        - data_path (str): Path to the data file.
        """
        self.df = pd.read_parquet(data_path, engine='pyarrow')

    def _get_weight_ffd(self, d, thres, lim):
        """
        Calculate the weights for fractional differentiation.

        Parameters:
        - d (float): Differentiation factor.
        - thres (float): Threshold for weight calculation.
        - lim (int): Limit for weight calculation.

        Returns:
        - np.ndarray: Array of weights.
        """
        w, k = [1.], 1
        ctr = 0
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
            ctr += 1
            if ctr == lim - 2:
                break
        w = np.array(w[::-1])
        return w

    def frac_diff_ffd(self, col_name, d, thres=1e-5, disable_warning=False):
        """
        Apply fractional differentiation to a specific column in the data.

        Parameters:
        - col_name (str): Column name for differentiation.
        - d (float): Differentiation factor.
        - thres (float): Threshold for weight calculation.
        - disable_warning (bool): Disable warning about applying log.

        Returns:
        - np.ndarray: Array of fractionally differentiated values.
        """
        # col_data = self.df[col_name].apply(np.log).values
        col_data = np.log(self.df[col_name].values)
        if np.max(col_data) > 10.0 and not disable_warning:
            print('WARNING: have you applied log before calling this function? If yes, discard this warning.')
        w = self._get_weight_ffd(d, thres, len(col_data))
        width = len(w) - 1
        output = np.zeros_like(col_data)
        for i in range(width, len(col_data)):
            output[i] = np.dot(w.T, col_data[i - width:i + 1])
        return output
    
# Example usage:
financial_analysis = FinancialAnalysis('../data/cryptos/BTCUSD_2020-04-07_2022-04-06.parquet')
result = financial_analysis.frac_diff_ffd(col_name='OPEN', d=0.4)
print(result)
