import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt


def load_data(file_path):
    real_estate_data = pd.read_parquet(file_path)
    return real_estate_data


def preprocess_data(data):
    features = ['VOLUME', 'VW', 'OPEN', 'CLOSE', 'HIGHT', 'LOW', 'TRANSACTIONS', 'DATE']

    # Ensure 'DATE' column is in Pandas datetime format
    data['DATE'] = pd.to_datetime(data['DATE'])

    # Convert 'DATE' column to NumPy array before subtraction
    data['DATE'] = (data['DATE'].values.astype(np.int64) - pd.Timestamp("1970-01-01").value) // int(
        1e9)  # Use int(1e9) instead of '1s'

    df_real_estate = data[features]
    return df_real_estate


def kmeans_silhouette_scores(data, k_values):
    silhouette_scores = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))

    return silhouette_scores


def kmeans_calinski_scores(data, k_values):
    calinski_scores = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        calinski_scores.append(calinski_harabasz_score(data, labels))

    return calinski_scores


def calculate_gap_statistic(data, labels, k):
    random_data = np.random.rand(*data.shape)
    random_labels = KMeans(n_clusters=k, random_state=42).fit_predict(random_data)

    real_dispersion = np.sum(
        np.min(pairwise_distances(data.iloc[:, :-1], data.iloc[labels, :-1], metric='euclidean'), axis=1)) / len(data)
    random_dispersion = np.sum(
        np.min(pairwise_distances(data.iloc[:, :-1], data.iloc[labels, :-1], metric='euclidean'), axis=1)) / len(data)

    return np.log(random_dispersion) - np.log(real_dispersion)



class SomeSpecificException:
    pass


def elbow_method_visualization(data, some_condition=None):
    try:
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(2, 30))
        visualizer.fit(data)
        visualizer.show()
        plt.title("Elbow Method for K means")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Distortion")
        plt.legend(loc='best')
        if len(data['clusters']) < 2:
            raise ValueError("Number of clusters should be at least 2")

    except ValueError as e:
    # Catch the specific exception and re-raise it
        raise e


def gap_statistic_visualization(data, k_values):
    gap_scores = [calculate_gap_statistic(data, labels, k) for k in k_values]

    plt.figure(figsize=(10, 4))
    plt.plot(k_values, gap_scores, marker='o', color='green', label='Gap Statistic')
    plt.title('Gap Statistic for K-Means Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Gap Statistic')
    plt.legend(loc='best')


if __name__ == "__main__":
    # Example usage of the functions
    file_path = 'dataset.parquet'
    real_estate_data = load_data(file_path)
    df_real_estate = preprocess_data(real_estate_data)

    k_values = range(2, 31)
    silhouette_scores = kmeans_silhouette_scores(df_real_estate, k_values)
    calinski_scores = kmeans_calinski_scores(df_real_estate, k_values)

    elbow_method_visualization(df_real_estate)
    gap_statistic_visualization(df_real_estate, k_values)
    hierarchical_dendrogram(df_real_estate)