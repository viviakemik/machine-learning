import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def show_pie_charts(portfolios: pd.DataFrame) -> None:
    if type(portfolios) != pd.DataFrame:
        raise ValueError(
            f'Function show_pie_charts - Parameter portfolios has to be of type'
            f'pandas.DataFrame, but got type {type(portfolios)}')

    how_many_portfolios = len(portfolios.columns)
    fig, ax = plt.subplots(1, how_many_portfolios, figsize=(30, 20))
    for i in range(how_many_portfolios):
        ax[i].pie(portfolios.iloc[:, i], )
        ax[i].set_title(portfolios.columns[i], fontsize=30)

    portfolios.plot.pie(subplots=True, figsize=(20, 10), legend=False)


def show_dendogram(link: np.ndarray or np.array, labels: tuple or list or pd.Index) -> None:
    if type(link) not in [np.ndarray, np.array]:
        raise ValueError(f'Function show_dendogram - Parameter link has to be of type '
                         f'numpy.array or numpy.ndarray, but got type {type(link)}')
    if type(labels) not in [tuple, list, pd.Index]:
        raise ValueError(f'Function show_dendogram - Parameter labels has to be of type '
                         f'tuple or list, but got type {type(labels)}')
    if link.shape[0] != len(labels) - 1:
        raise ValueError(
            f'Function show_dendogram - Parameter link has to contain countwise '
            f'columns = given labels - 1, but link has shape {link.shape} '
            f'and labels contains {len(labels)} elements, {link.shape[0]} != {len(labels) - 1}.')
    # Plot Dendogram
    plt.figure(figsize=(20, 7))
    plt.title("Dendrogram")
    dendrogram(link, labels=labels)
    plt.show()


def show_result_plot(result_df: pd.DataFrame, title: str = "") -> None:
    if type(result_df) != pd.DataFrame:
        raise ValueError(
            f'HierarchRiskParity Class - method show_result_plot - Parameter result_df has to be of type'
            f'pandas.DataFrame, but got type {type(result_df)}')
    if type(result_df) != pd.DataFrame:
        raise ValueError(
            f'HierarchRiskParity Class - method show_result_plot - Parameter title has to be of type'
            f'str, but got type {type(title)}')
    result_df.cumsum().plot(figsize=(10, 5), title=title)
