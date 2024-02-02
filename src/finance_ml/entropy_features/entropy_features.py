"""
Created on Fri, Jan 26 2024
@Group: 
    Abhinav Choudhary
    Ankit Joshi
    Sejal Bangar
    Vaishnav Negi
"""

import numpy as np
from scipy.stats import entropy
 
class EntropyFeatures:
    def __init__(self):
        pass
 
    def plugin_entropy_estimator(self, data):
        """
        Calculate the Plug-in Entropy Estimator for each column in the financial dataset.
 
        Returns:
        - dict: Dictionary with feature names as keys and their corresponding entropy values.
        """
        entropy_values = {}
 
        # Check if data is loaded
        if data is None:
            raise ValueError("Data has not been loaded. Use load_data method before calculating entropy.")
 
        # Iterate over each column in the DataFrame
        for column in data.columns:
            # Extract the values of the current column
            current_column_values = data[column].values
 
            # Discretize the continuous variable into bins
            bins = 10
            discretized_values = np.digitize(current_column_values, bins=np.linspace(min(current_column_values), max(current_column_values), bins + 1))
 
            # Calculate the empirical probability distribution
            empirical_probs = np.histogram(discretized_values, bins=bins, density=True)[0]
 
            # Calculate the entropy using the Plug-in Entropy Estimator
            entropy_values[column] = entropy(empirical_probs, base=2)
 
        return entropy_values
 
    def lempel_ziv_estimator(self, data):
        """
        Calculate the Lempel-Ziv Entropy Estimator for each column in the financial dataset.
 
        Returns:
        - dict: Dictionary with feature names as keys and their corresponding Lempel-Ziv entropy values.
        """
        entropy_values = {}
 
        # Check if data is loaded
        if data is None:
            raise ValueError("Data has not been loaded. Use load_data method before calculating entropy.")
 
        # Iterate over each column in the DataFrame
        for column in data.columns:
            # Convert the column to a sequence of strings
            current_sequence = data[column].astype(str).str.cat(sep=' ')
 
            # Initialize the dictionary for the LZ compression
            dictionary = {current_sequence[0]: 0}
            code = 1
            current_code = 0
 
            # Count the number of distinct patterns in the sequence
            for char in current_sequence[1:]:
                new_sequence = current_sequence[current_code] + char
                if new_sequence not in dictionary:
                    dictionary[new_sequence] = code
                    code += 1
                    current_code += 1
 
            # Calculate the Lempel-Ziv entropy
            entropy_values[column] = len(dictionary)
 
        return entropy_values
