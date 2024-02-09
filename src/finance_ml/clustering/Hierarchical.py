import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage

__all__ = [
    "Hierarchical",
]

class Hierarchical:
    def __init__(self, dataset: pd.DataFrame, labels: list, interval: int = 24):
        '''
        Initialize the Hierarchical object for stock data analysis.

        Args:
            dataset: pandas.DataFrame
                The dataset containing stock prices or returns, indexed by time.
            labels: list of str
                The labels for each column in the dataset, typically stock names.
            interval: int, optional
                The window interval for sliding window calculations. Defaults to 24.
                Set to less than 1 to skip sliding window in preprocessing.
        '''
        self.interval:int = interval
        self.dataset:pd.DataFrame = dataset  # Dataset should be indexed by time.
        self.labels:list = labels if labels is not None else dataset.columns.tolist()

        # DataFrames to store calculated returns and standard deviations (risks)
        self.returns_df:pd.DataFrame = None
        self.segment_std_df:pd.DataFrame = None

        # Data linkage for dendrogram
        self.data_linkage = None

    def preprocessing(self, fillna_method: str = '', resample: str = 'H', inplace: bool = True, standardize_returns: bool = True, standardize_risks: bool = True):
        '''
        Preprocesses the dataset by handling missing values, removing duplicates,
        applying sliding window, and standardizing data.

        Args:
            fillna_method: str, optional
                Method for handling missing values. Options: 'dropna', 'ffill', 'bfill', or a numeric value for fixed fill.
            resample: str, optional
                Frequency for resampling the data. Defaults to 'H' (hourly).
            inplace: bool, optional
                Whether to modify the dataset in place. Defaults to True.
            standardize_returns: bool, optional
                Whether to standardize returns data. Defaults to True.
            standardize_risks: bool, optional
                Whether to standardize risks data. Defaults to True.
        '''

        # Handling missing values
        if fillna_method == 'dropna':
            self.dataset.dropna(inplace=inplace)
        elif fillna_method in ['ffill', 'bfill']:
            self.dataset.fillna(method=fillna_method, inplace=inplace)
        elif fillna_method.isnumeric():
            self.dataset.fillna(float(fillna_method), inplace=inplace)
        else:
            # Default option if fillna_method is not specified
            self.dataset.drop_duplicates(inplace=inplace)
            self.dataset = self.dataset.resample(resample).interpolate()
            self.dataset.dropna(inplace=True)

        # Apply sliding window if interval is set
        if self.interval > 1:
            self.apply_sliding_window(self.dataset, self.interval)

        # Standardize the data
        if standardize_returns:
            self.returns_df = self.standardize_dataset(self.returns_df)
        if standardize_risks:
            self.segment_std_df = self.standardize_dataset(self.segment_std_df)

    def apply_sliding_window(self, dataset: pd.DataFrame, interval: int = 24):
        '''
        Applies a sliding window to the dataset and calculates returns and risks for each window.

        Args:
            dataset: pandas.DataFrame
                The dataset on which the sliding window is to be applied.
            interval: int, optional
                The window interval for sliding window calculations. Defaults to 24.
        '''

        # Initialize DataFrames to store the results
        self.returns_df = pd.DataFrame(index=range(dataset.shape[0] - interval + 1))
        self.segment_std_df = pd.DataFrame(index=range(dataset.shape[0] - interval + 1))

        # Apply sliding window and calculate returns and standard deviations for each column
        for column in dataset.columns:
            segmented = sliding_window_view(dataset[column], interval, axis=0)
            if segmented.size == 0:
                continue  # Skip if the segmented window is empty

            # Calculate returns and standard deviation (risk) for each segment
            segment_returns = (segmented[:, -1] - segmented[:, 0]) / segmented[:, 0]
            segment_std = np.std(segmented, axis=1)

            # Store the calculated values in respective DataFrames
            self.returns_df[column] = segment_returns
            self.segment_std_df[column] = segment_std

    def standardize_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        '''
        Standardizes the given dataset using MinMaxScaler.

        Args:
            dataset: pandas.DataFrame
                The dataset to be standardized.

        Returns:
            pandas.DataFrame
                The standardized dataset.
        '''
        scaler = MinMaxScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns, index=dataset.index)
        return scaled_df

    def find_linkage(self, df: pd.DataFrame, plot_dendogram: bool = True, method: str = 'ward',
                     x_label: str = 'Returns'):
        '''
        Calculates and optionally plots a dendrogram based on hierarchical clustering.

        Args:
            df: pandas.DataFrame
                The DataFrame to be used for hierarchical clustering.
            plot_dendogram: bool, optional
                If True, plots the dendrogram. Defaults to True.
            method: str, optional
                The linkage algorithm to use. Defaults to 'ward'.
            x_label: str, optional
                Label for the x-axis in the dendrogram plot. Defaults to 'Returns'.
        '''
        # Calculate the linkage for hierarchical clustering
        returns_scaled = df.transpose().to_numpy()
        self.data_linkage = linkage(returns_scaled, method=method)

        # Plot the dendrogram if requested
        if plot_dendogram:
            plt.figure(figsize=(10, 7))
            dendrogram(self.data_linkage, labels=self.labels, orientation='top', distance_sort='descending',
                       show_leaf_counts=True)
            plt.title("Dendrogram ({method}'s Method)".format(method=method))
            plt.xlabel("Hierarchical Clustering of Standardized " + x_label)
            plt.ylabel("Distance")
            plt.xticks(rotation=45)
            plt.show()

    def plot_stocks_timeline(self, dataset: pd.DataFrame = None, y_label: str = 'returns', show_legends: bool = True):
        '''
        Plots the timeline of stock data.

        Args:
            dataset: pandas.DataFrame, optional
                The dataset to plot. If None, uses self.returns_df.
            y_label: str, optional
                Label for the y-axis. Defaults to 'returns'.
            show_legends: bool, optional
                Whether to show the legend. Defaults to True.
        '''
        # Use self.returns_df if no dataset is provided
        if dataset is None:
            dataset = self.returns_df.copy()

        # Plot each stock's timeline
        plt.figure(figsize=(10, 7))
        for column in dataset.columns:
            plt.plot(dataset.index, dataset[column], label=column)

        # Add legends, title, and labels
        if show_legends:
            plt.legend()
        plt.title('Stock Prices Over Time')
        plt.xlabel('Time')
        plt.ylabel(y_label)
        plt.show()

    def plot_stocks_avg_risk(self):
        '''
        Plots the average risk (standard deviation) for each stock.
        '''
        mean_risks = self.segment_std_df.mean()
        self._bar_plot(mean_risks, self.labels, 'Stocks', 'Risk', 'Average Risk per Stock')


    def plot_stocks_avg_return(self):
        '''
        Plots the average return for each stock.
        '''
        mean_returns = self.returns_df.mean()
        self._bar_plot(mean_returns, self.labels, 'Stocks', 'Return', 'Average Return per Stock')


    def _bar_plot(self, data: pd.Series, labels: list, x_label: str, y_label: str, title: str):
        '''
        Helper function to plot a bar chart.

        Args:
            data: pandas.Series
                The data to plot.
            labels: list
                The labels for each bar.
            x_label: str
                The label for the x-axis.
            y_label: str
                The label for the y-axis.
            title: str
                The title of the plot.
        '''
        # Plotting the bar chart
        plt.figure(figsize=(10, 7))
        plt.bar(labels, data, color='blue')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.show()

    def plot_stocks_risk_vs_return(self, plot_sharpe_ratio: bool = False, risk_tolerance: float = 0.02,
                                   expected_return: float = 0.1, risk_free_rate: float = 0.02):
        '''
        Plots the risk vs return of stocks, optionally with the Capital Market Line.

        Args:
            plot_sharpe_ratio: bool, optional
                If True, plots the Capital Market Line based on the Sharpe ratio. Defaults to False.
            risk_tolerance: float, optional
                The standard deviation of the portfolio. Used for Sharpe ratio calculation. Defaults to 0.02.
            expected_return: float, optional
                The expected return of the portfolio. Used for Sharpe ratio calculation. Defaults to 0.1.
            risk_free_rate: float, optional
                The risk-free rate. Used for Sharpe ratio calculation. Defaults to 0.02.
        '''
        # Calculate and plot risk vs return for each stock
        mean_returns = self.returns_df.mean() * 100
        mean_risks_abs = self.segment_std_df.abs().mean() * 100
        plt.figure(figsize=(10, 7))
        for index, label in enumerate(self.labels):
            plt.scatter(mean_risks_abs[index], mean_returns[index])
            plt.text(mean_risks_abs[index], mean_returns[index], label, fontsize=9, ha='right')

        # Plot the Capital Market Line if requested
        if plot_sharpe_ratio:
            sharpe_ratio = (expected_return - risk_free_rate) / risk_tolerance
            x = np.linspace(min(mean_risks_abs), max(mean_risks_abs), 100)
            y = sharpe_ratio * x + risk_free_rate
            plt.plot(x, y, label=f"Capital Market Line \nSharpe Ratio: {sharpe_ratio:.2f}"
                                 f"\nRisk Free Rate: {risk_free_rate:.2f}\nExpected Return: {expected_return:.2f}"
                                 f"\nRisk Tolerance: {risk_tolerance:.2f}")

        plt.xlabel('Risk% (Standard Deviation)')
        plt.ylabel('Returns%')
        plt.title('Risk vs Return')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.show()
