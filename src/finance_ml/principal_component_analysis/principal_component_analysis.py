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


    def _init_(self, data: pd.DataFrame):
        '''
        Initializes the PrincipalComponentAnalysis class and standardizes the dataset for PCA analysis.
        Args:
        -----
        data : pd.DataFrame
            The dataset to be analyzed using PCA. This should be a Pandas DataFrame containing the data that will be standardized as a preprocessing step for PCA.
            Standardizes the dataset as a crucial preprocessing step for PCA analysis. This method uses the StandardScaler from sklearn.preprocessing to transform the dataset such that each feature has a mean of 0 and a standard deviation of 1.
        Returns:
        -----
        None
            This method does not return any value but standardizes the dataset in preparation for PCA analysis.
        '''

        self.data = data
        self.pca = None
        self.pca_result = None
        
    def preprocess_data(self) -> pd.DataFrame:

        '''
            Standardizes the dataset to have a mean of 0 and standard deviation of 1 before applying PCA. 
            This ensures that each feature contributes equally to the analysis.
            Args:
            -----
            data : pd.DataFrame
                The dataset to be standardized. Assumes that data is a Pandas DataFrame where rows represent samples and columns represent features.
            Returns:
            -----
            standardized_data: pd.DataFrame
                A DataFrame where the data has been standardized to have a mean of 0 and standard deviation of 1 for each feature.
        '''

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        return scaled_data

    def apply_pca(self, data):
        '''
            Apply PCA on the preprocessed data to reduce dimensionality and capture significant variance.
            Args:
            -----
            n_components : int
                The number of principal components to keep. In this context, n_components=2, indicating that the dataset should be reduced to two principal components.
            preprocessed_data : np.ndarray or pd.DataFrame
                The preprocessed data on which PCA is to be applied. This data should already be standardized or normalized.
            Returns:
            -----
            self.pca_result : np.ndarray
                The transformed dataset represented in terms of its principal components. This output contains the original data projected into the PCA space defined by the two most significant principal components.
        '''

        self.pca = PCA(n_components=2)
        self.pca_result = self.pca.fit_transform(self.preprocess_data())
        data = self.pca.fit(data)
        return data
    
    def plot_explained_variance_component(self, pca):

        '''
            Short description:
            ------------------
            This function visualizes the explained variance ratio of each principal component derived from a PCA (Principal Component Analysis) model. It helps understand how much variance each principal component accounts for in the dataset.
            Args:
            -----
            pca: PCA object
                The fitted PCA object from sklearn.decomposition.PCA. It contains information about the principal components, including the explained variance ratio.
            Returns:
            -----
            None
                This function does not return any value. It generates a plot showing the explained variance ratio for each principal component.
        '''

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

        '''
            Visualizes the PCA results using a scatter plot. This function plots the first two principal components of the dataset after PCA transformation, allowing for the visualization of the dataset in a reduced dimensionality space.
            Args:
            -----
            colour : data type (likely a sequence type compatible with matplotlib, e.g., list or array)
                The color mapping for each data point in the scatter plot, allowing for differentiation based on some criterion.
            Returns:
            -----
            None : This function does not return a value but shows a matplotlib plot directly.
        '''

        if self.pca_result is None:
            raise ValueError("PCA has not been applied yet.")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.pca_result[:, 0], y=self.pca_result[:, 1], c=colour)
        plt.title(f'PCA Results with 2 Components')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    def explained_variance(self):

        '''
            Short description:
                This method prints the explained variance ratio for each principal component after applying Principal Component Analysis (PCA) to a dataset. 
                It checks if PCA has been applied (self.pca is not None) and raises an exception if not, indicating the PCA model must be fit to the data before accessing the explained variance ratios.
            Args:
            -----
            None
            Raises:
            -----
            Exception: If self.pca is None, indicating that PCA has not been applied yet and the PCA model is not fit to the data.
            Returns:
            -----
            None
        '''

        if self.pca is None:
            raise ValueError("PCA has not been applied yet.")

        for i, variance in enumerate(self.pca.explained_variance_ratio_):
            print(f"Principal Component {i+1}: {variance:.2f}")

    def correlation_matrix(self, data):

        '''
            Creates a heatmap visualization of the correlation matrix for a dataset.
            Args:
            -----
            data : DataFrame
                The dataset for which the correlation matrix and its heatmap visualization are to be generated.
            Returns:
            -----
            A heatmap plot is displayed, illustrating the correlation matrix of the dataset's features. This function does not return a value but visualizes the correlation matrix using matplotlib.
            Note:
            -----
            The correlation values range from -1 to 1, where -1 indicates a perfect negative correlation, 0 indicates no correlation, and 1 indicates a perfect positive correlation. The colormap 'jet' is used to represent this range, with different colors indicating the strength and direction of correlations.
        '''

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

