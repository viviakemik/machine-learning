import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

__all__ = [
    "Hierarchical",
]


class Hierarchical:
    def __init__(self, dataset, labels, interval=24):
        '''
        Initialize the Hierarchical object for stock data analysis.

        Args:
        -----
        dataset : pandas.DataFrame
            The dataset containing stock prices or returns.
        labels : list of str
            The labels for each column in the dataset, typically stock names.
        interval : int, optional
            The window interval for sliding window calculations, defaults to 24.
        '''

        self.interval = interval
        self.dataset = dataset
        self.labels = labels

        # DataFrames to store various calculated values
        self.returns_df = None
        self.segment_std_df = None
        self.sharpe_ratio_df = None

        # Dendrogram related information
        self.data_linkage = None

    def preprocessing(self, fillna_method='dropna', inplace=True,
                      standardize_returns=True, standardize_risks=True, standardize_sharpe_ratio=True):
        '''
        Preprocesses the dataset by handling missing values, removing duplicates,
        applying sliding window, and standardizing data.

        Args:
        -----
        fillna_method : str, optional
            Method for handling missing values. Options are 'dropna' (default),
            'ffill', 'bfill', or a numeric value for fixed fill.
        inplace : bool, optional
            Whether to modify the dataset in place. Defaults to True.
        standardize_returns : bool, optional
            Whether to standardize returns data. Defaults to True.
        standardize_risks : bool, optional
            Whether to standardize risks data. Defaults to True.
        standardize_sharpe_ratio : bool, optional
            Whether to standardize Sharpe ratio data. Defaults to True.
        '''

        self.dataset.drop_duplicates(inplace=inplace)
        if fillna_method == 'dropna':
            self.dataset.dropna(inplace=inplace)
        elif fillna_method in ['ffill', 'bfill']:
            self.dataset.fillna(method=fillna_method, inplace=inplace)
        elif fillna_method.isnumeric():
            self.dataset.fillna(method=int(fillna_method), inplace=inplace)

        if self.interval > 1:
            self.apply_sliding_window(self.dataset, self.interval)

        if standardize_returns:
            self.returns_df = self.standardize_dataset(self.returns_df)
        if standardize_risks:
            self.segment_std_df = self.standardize_dataset(self.segment_std_df)
        if standardize_sharpe_ratio:
            self.sharpe_ratio_df = self.standardize_dataset(self.sharpe_ratio_df)

    def apply_sliding_window(self, dataset, interval=24):
        '''
        Applies a sliding window to the dataset and calculates returns, risks,
        and Sharpe ratios for each window.

        Args:
        -----
        dataset : pandas.DataFrame
            The dataset on which the sliding window is to be applied.
        interval : int, optional
            The window interval for sliding window calculations, defaults to 24.
        '''

        self.returns_df = pd.DataFrame(index=range(dataset.shape[0] - interval + 1))
        self.sharpe_ratio_df = pd.DataFrame(index=range(dataset.shape[0] - interval + 1))
        self.segment_std_df = pd.DataFrame(index=range(dataset.shape[0] - interval + 1))

        for column in dataset.columns:
            segmented = sliding_window_view(dataset[column], interval, axis=0)
            if segmented.size == 0:
                continue
            segment_returns = segmented[:, 0] - segmented[:, -1]
            segment_std = np.std(segmented, axis=1)
            sharpe_ratio = np.divide(segment_returns, segment_std, out=np.zeros_like(segment_returns),
                                     where=segment_std != 0)

            self.returns_df[column] = segment_returns
            self.segment_std_df[column] = segment_std
            self.sharpe_ratio_df[column] = sharpe_ratio

    def standardize_dataset(self, dataset):
        '''
        Standardizes the given dataset using StandardScaler.

        Args:
        -----
        dataset : pandas.DataFrame
            The dataset to be standardized.

        Returns:
        -----
        pandas.DataFrame
            The standardized dataset.
        '''

        scaler = StandardScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns, index=dataset.index)
        return scaled_df


    def find_linkage(self, df, plot_dendogram=True, method='ward'):
        ## todo find and add other methods
        plt.figure(figsize=(20, 12))
        returns_scaled = df.transpose().to_numpy()
        self.data_linkage = linkage(returns_scaled, method='complete')
        # Plotting the Dendrogram
        if plot_dendogram:
            plt.figure()
            dendrogram(self.data_linkage, labels=self.labels, orientation='top',
                       distance_sort='descending', show_leaf_counts=True)
            plt.title("Distance ({method}'s Method)".format(method=method))
            plt.xlabel("Hierarchical Clustering of Standardized Returns")
            plt.ylabel("Stocks")
            plt.xticks(rotation=45)
            plt.show()

    def plot_stocks_timeline(self, dataset=None, y_label='returns', show_legends=True):
        # Plotting each column
        if dataset is None:
            dataset = self.returns_df.copy()
        for column in dataset.columns:
            plt.plot(dataset.index, dataset[column], label=column)

        if show_legends:
            plt.legend()  # Adding legends
        # Adding title and labels (modify as needed)
        plt.title('Stock Prices Over Time')
        plt.xlabel('Time')
        plt.ylabel(y_label)
        plt.show()  # Show the plot

    def plot_stocks_risk_bar_plot(self):
        plt.figure()
        mean_risks = self.segment_std_df.mean()
        plt.bar(self.labels, mean_risks, color='blue')
        plt.xlabel('Stocks')
        plt.ylabel('Risk')
        plt.title('Risk plot')
        plt.xticks(rotation=45)
        plt.show()

    def plot_stocks_return_bar_plot(self):
        plt.figure()
        mean_returns = self.returns_df.mean()
        plt.bar(self.labels, mean_returns, color='blue')
        plt.xlabel('Stocks')
        plt.ylabel('Returns')
        plt.title('Return plot')
        plt.xticks(rotation=45)
        plt.show()

    def plot_stocks_risk_vs_return_plot(self):
        plt.figure()
        mean_returns = self.returns_df.mean()
        mean_risks = self.segment_std_df.mean()
        for label in mean_returns.index:
            plt.scatter(mean_risks[label], mean_returns[label])
            plt.text(mean_risks[label], mean_returns[label], label, fontsize=9, ha='right')

        plt.xlabel('Risks')
        plt.ylabel('Returns')
        plt.title('Risk vs Return')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.show()


    def plot_stocks_sherpe_ratio(self):
        plt.figure()
        mean_sherpe_ratio = self.sherpe_ratio_df.mean()
        plt.bar(self.labels, mean_sherpe_ratio, color='blue')
        plt.xlabel('Stocks')
        plt.ylabel('Sherpe Ratio')
        plt.title('Return plot')
        plt.xticks(rotation=45)
        plt.show()
