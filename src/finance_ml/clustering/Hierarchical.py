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
        # if interval is set to 1 or less, sliding window is skipped
        self.interval = interval
        self.dataset = dataset
        self.labels = labels

        # After applying sliding window
        self.returns_df = None
        self.segment_std_df = None
        self.sherpe_ratio_df = None

        # dendogram related information
        self.data_linkage = None

    def preprocessing(self, fillna_with='dropna', inplace=True,
                      standardized_returns=True, standardized_risks=True, standardized_sherpe_ratio=True):
        # possible values for fillna is ffill - forward fill , bfill - backward fill, 0 - for fixed fill
        # Handling NaNs
        # More advanced duplicate handling might require custom logic
        # dropping na will give the most performance if we are using the complete asset information
        # fillna may creating issue while assigning memory, as memory requirement may go in 10s of TB
        # df.fillna(method='ffill', inplace=True)  # Forward fill
        # df.fillna(method='bfill', inplace=True)  # Backward fill
        # df.fillna(0, inplace=True)               # Fill with a specific value like 0
        # df.interpolate(inplace=True)             # Interpolation
        # Handling duplicates

        self.dataset.drop_duplicates(inplace=inplace)

        print(self.dataset.shape)
        if fillna_with == 'dropna':
            self.dataset.dropna(inplace=inplace)
        elif fillna_with == 'ffill' or fillna_with == 'bfill':
            self.dataset.fillna(method=fillna_with, inplace=inplace)
        elif fillna_with.isnumeric():
            self.dataset.fillna(method=int(fillna_with), inplace=inplace)

        ## todo implement resample
        ## todo give another option to set the duplicates with group by mean value
        if self.interval > 1:
            self.apply_sliding_window(self.dataset, self.interval)

        if standardized_returns:
            self.returns_df = self.standardized_dataset(self.returns_df)
        if standardized_risks:
            self.segment_std_df = self.standardized_dataset(self.segment_std_df)
        if standardized_sherpe_ratio:
            self.sherpe_ratio_df = self.standardized_dataset(self.sherpe_ratio_df)

    def apply_sliding_window(self, dataset, interval=24):
        # Create a copy of the DataFrame to store results
        # This copy will have fewer rows lens = t.rows - interval + 1
        self.returns_df = pd.DataFrame(index=range(self.dataset.shape[0] - interval + 1))
        self.sherpe_ratio_df = pd.DataFrame(index=range(self.dataset.shape[0] - interval + 1))
        self.segment_std_df = pd.DataFrame(index=range(self.dataset.shape[0] - interval + 1))

        # Iterate through each column in the DataFrame
        for column in self.dataset.columns:
            # Apply sliding window
            segmented = sliding_window_view(self.dataset[column], interval, axis=0)

            if segmented.size == 0:
                continue

            # Calculating the returns in each segment
            segment_returns = segmented[:, 0] - segmented[:, -1]
            self.returns_df[f'{column}'] = segment_returns

            # Calculating risk (standard deviation) in each segment
            segment_std = np.std(segmented, axis=1)
            self.segment_std_df[f'{column}'] = segment_std

            # Calculating Sharpe ratios
            sharpe_ratio = np.divide(segment_returns, segment_std, out=np.zeros_like(segment_returns),
                                     where=segment_std != 0)
            self.sherpe_ratio_df[f'{column}'] = sharpe_ratio

    def standardized_dataset(self, dataset):
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
