"""
Created on Wednesday, 17 January 2024
@Group:
    Siddharth Vijay Mane (un14afyz)
    Akshat Anand Khara (mi77sopu)
    Prathamesh Shankar Agare (da92pita)
    Muskan Muskan (uk98etil)
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class PrincipalComponentAnalysis:
    # ----------------------------------------------------------------------------------
    # Class to perform principal component analysis in a dataset.
    # ----------------------------------------------------------------------------------

    def __init__(self, data: pd.DataFrame):
        # I.1 Initialize the PrincipalComponentAnalysis class with data.
        # I.1.1:param data: Pandas DataFrame containing the data.
        self.data = data
        self.pca = None
        self.pca_result = None
        
    def preprocess_data(self) -> pd.DataFrame:
        # I.2 Standardize the data before applying PCA.
        # I.2.1 :return: Standardized DataFrame.
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        return scaled_data

    def apply_pca(self, data):
        # I.3 Apply PCA on the preprocessed data.
        self.pca = PCA(n_components=2)
        self.pca_result = self.pca.fit_transform(self.preprocess_data())
        data = self.pca.fit(data)
        return data
    
    def plot_explained_variance_component(self, pca):
        plt.figure(figsize=(10,6))
        # Number of components
        num_components = len(pca.explained_variance_ratio_)
        plt.scatter(x=[i+1 for i in range(len(pca.explained_variance_ratio_))],
        y=pca.explained_variance_ratio_,
        s=200, alpha=0.75,c=plt.cm.rainbow(np.linspace(0, 1, num_components)) ,edgecolor='k')
        plt.grid(True)
        plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=25)
        plt.xlabel("Principal components",fontsize=15)
        plt.xticks([i+1 for i in range(len(pca.explained_variance_ratio_))],fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("Explained variance ratio",fontsize=15)
        plt.show()

    def plot_pca_results(self, colour):
        # I.4 Visualize the PCA results using a scatter plot.
        if self.pca_result is None:
            raise ValueError("PCA has not been applied yet.")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.pca_result[:, 0], y=self.pca_result[:, 1], c=colour)
        plt.title(f'PCA Results with 2 Components')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    def explained_variance(self):
        # Print the explained variance ratio of each principal component.
        if self.pca is None:
            raise ValueError("PCA has not been applied yet.")

        for i, variance in enumerate(self.pca.explained_variance_ratio_):
            print(f"Principal Component {i+1}: {variance:.2f}")

    def correlation_matrix(self, data):
        from matplotlib import pyplot as plt
        from matplotlib import cm as cm

        fig = plt.figure(figsize=(16,12))
        ax1 = fig.add_subplot(111)
        cmap = cm.get_cmap('jet', 30)
        cax = ax1.imshow(data.corr(), interpolation="nearest", cmap=cmap)
        ax1.grid(True)
        plt.title('Features correlation\n',fontsize=15)
        labels=data.columns
        ax1.set_xticklabels(labels,fontsize=9)
        ax1.set_yticklabels(labels,fontsize=9)
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
        plt.show()