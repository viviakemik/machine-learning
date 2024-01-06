"""
Created on Wed Dec 20, 2023

@authors: Sebastian Kreuz, Maximilian Ehmann

The content is based on the case study of Hariom Tatsath and Sahil Puri which can be found in
https://github.com/tatsath/fin-ml/tree/master/Chapter%208%20-%20Unsup.%20Learning%20-%20Clustering/Case%20Study3%20-%20Hierarchial%20Risk%20Parity

"""
# Import required packages
from typing import Callable
import numpy as np
import pandas as pd
import pandas.core.frame

# Import Model Packages
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

# Package for optimization of mean variance optimization
import cvxopt as opt
from cvxopt import blas, solvers


class PortfolioOptimization:
    def __init__(self, data_train: pd.core.frame.DataFrame, data_test: pd.core.frame.DataFrame,
                 correl_dist: Callable[[pd.core.frame.DataFrame], pd.core.frame.DataFrame] = lambda corr: (
                                                                                                                  (
                                                                                                                          1 - corr) / 2.) ** .5,
                 drop_null: bool = False):
        """
        Initialize data for portfolio optimization currently including
            - Hierarchical Risk Parity (HRP)
            - Inverse Variance Portfolio (IVP)
            - Critical Line Algorithm (CLA)

        Args:
            data_train (pandas DataFrame): The data that should be used for training,
            data_test (pandas DataFrame): The data that should be used for testing,
            optional: correl_dist(function (pandas DataFrame) -> pandas DataFrame): The distance that should be used
                                for correlation matrix, defaults to sqrt((1-corr)/2)
            optional: drop_null(bool): If rows with null/n.a. values in the data should be discarded. If False but null/n.a.
                                value exists in the data, an error occurs. You should consider cleaning the data before
                                calling this function.
        """
        # First check that all the inputs are in the correct formats and don't contain null/n.a. values
        if type(data_train) != pd.core.frame.DataFrame:
            raise ValueError(f'HierarchRiskParity Class - Parameter data_train must be pandas.core.frame.DataFrame,'
                             f' got type {type(data_train)}')
        if type(data_test) != pd.core.frame.DataFrame:
            raise ValueError(f'HierarchRiskParity Class - Parameter data_test must be pandas.core.frame.DataFrame,'
                             f' got type {type(data_test)}')
        if not callable(correl_dist):
            raise ValueError(f'HierarchRiskParity Class - Parameter function correl_dist must be callable,'
                             f' got type {type(correl_dist)}')
        df_test = pd.DataFrame({
            'Age': [25, 30, 22, 35, 28],
            'Salary': [60000, 80000, 55000, 90000, 70000]
        })
        if type(correl_dist(df_test)) != pd.core.frame.DataFrame:
            raise ValueError(f'HierarchRiskParity Class - Parameter function correl_dist must return object of type,'
                             f'pandas.core.frame.DataFrame, got type {type(correl_dist(df_test))}')

        if type(drop_null) != bool:
            raise ValueError(f'HierarchRiskParity Class - Parameter drop_null must be bool,'
                             f' got type {type(drop_null)}')

        if drop_null:  # Drop all rows that have a null/n.a. value in them
            data_train = data_train.dropna()
            data_test = data_test.dropna()

        if data_train.isnull().values.any() or data_test.isnull().values.any():
            raise ValueError('HierarchRiskParity Class - Parameter data_train or data_test must not have null or N.A.'
                             'values if drop_null is False, consider cleaning the data before in your specificly'
                             'required setting, or drop all rows containing n.a. values by setting drop_null = True.')
        self.data_train = data_train
        self.data_test = data_test
        self.correl_dist = correl_dist
        self.drop_null = drop_null

        # Now compute additional parameters
        # Calculate percentage return
        returns_train = data_train.pct_change().dropna()
        returns_test = data_test.pct_change().dropna()
        if type(returns_train) != pd.core.frame.DataFrame:
            raise ValueError(
                f'HierarchRiskParity Class - Calculated parameter returns_train must be pandas.core.frame.DataFrame,'
                f' got type {type(returns_train)}')
        if type(returns_test) != pd.core.frame.DataFrame:
            raise ValueError(
                f'HierarchRiskParity Class - Calculated parameter returns_test must be pandas.core.frame.DataFrame,'
                f' got type {type(returns_test)}')

        # Calculate covariance and correlation
        cov, corr = returns_train.cov(), returns_train.corr()
        if type(cov) != pd.core.frame.DataFrame:
            raise ValueError(f'HierarchRiskParity Class - Calculated parameter cov must be pandas.core.frame.DataFrame,'
                             f' got type {type(cov)}')
        if type(corr) != pd.core.frame.DataFrame:
            raise ValueError(
                f'HierarchRiskParity Class - Calculated parameter corr must be pandas.core.frame.DataFrame,'
                f' got type {type(corr)}')

        self.returns_train = returns_train
        self.returns_test = returns_test
        self.cov = cov
        self.corr = corr

    def calc_linkage(self, method: str) -> np.ndarray:
        """
            Calculate the linkage which calculates the cluster tree by comparing and optimizing distances
            between different stocks

            Args:
                method (str): Parameter passed down to the linkage function from scipy, e.g. 'single'
        """
        if type(method) != str:
            raise ValueError(f'HierarchRiskParity Class - Calculated parameter method must be str,'
                             f' got type {type(method)}')
        # Convert the pandas dataframe output of self.correl_dist to a condensed distance matrix
        condensed_corr_dist = pdist(self.correl_dist(self.corr))
        return sch.linkage(condensed_corr_dist, method)

    def get_quasi_diag(self) -> list:
        """
            Get the quasi diagonalized linking/clustering matrix for hierarchical risk parity (HRP). This reordering of
            the assets places similar assets together and dissimilar assets are placed far apart.

            Returns a list of indices.
        """
        link = self.calc_linkage('single')
        # Sort clustered items by distance
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # number of original items
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
            df0 = sortIx[sortIx >= numItems]  # find clusters
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = pd.concat([sortIx, df0], ignore_index=True)  # item 2
            sortIx = sortIx.sort_index()  # re-sort
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()

    def get_cluster_var(self, c_items: list or pd.Index) -> np.ndarray:
        """
            Sub-method of the recursive bisection of the hierarchical risk parity (HRP). Takes a list of indices of a
            cluster and calculates and returns its variance.

            Args:
                c_items (list or pandas Index): Indices of a cluster, "cluster_items"
        """
        if type(c_items) not in [list, pd.Index]:
            raise ValueError(f'HierarchRiskParity Class - method get_cluster_var - Parameter cItems has to be of type '
                             f'list or pandas.core.indexes.base.Index, but got type {type(c_items)}')
        # Compute variance per cluster
        cov_ = self.cov.loc[c_items, c_items]  # matrix slice
        w_ = self.get_IVP(cov_).array.reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    def get_rec_bipart(self, sort_ix: list or pd.Index) -> pd.Series:
        """
            Perform the recursive bisection of the hierarchical risk parity (HRP). HRP now takes advantage of the
            quasi-diagonalization to calculate the individual asset weights.

            Args:
                sort_ix (list or pandas Index): The sorted indices from the quasi diagonalization.
        """
        if type(sort_ix) not in [list, pd.Index]:
            raise ValueError(f'HierarchRiskParity Class - method get_rec_bipart - Parameter sortIx has to be of type '
                             f'list or pandas.core.indexes.base.Index, but got type {type(sort_ix)}')
        # Compute HRP alloc
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]  # initialize all items in one cluster
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if
                       len(i) > 1]  # bi-section
            for i in range(0, len(c_items), 2):  # parse in pairs
                c_items0 = c_items[i]  # cluster 1
                c_items1 = c_items[i + 1]  # cluster 2
                c_var0 = self.get_cluster_var(c_items0)
                c_var1 = self.get_cluster_var(c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha  # weight 1
                w[c_items1] *= 1 - alpha  # weight 2
        return w

    def get_CLA(self) -> pd.Series:
        """
            Calculate and return the portfolio that is optimal in terms of the critical line algorithm (CLA).
        """
        N = 100
        mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

        # Convert to cvxopt matrices
        cov_as_array = self.cov.T.values
        n = len(cov_as_array)
        S = opt.matrix(cov_as_array)
        pbar = opt.matrix(np.ones(n))

        # Create constraint matrices
        G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
        h = opt.matrix(0.0, (n, 1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)

        # Calculate efficient frontier weights using quadratic programming
        solvers.options['show_progress'] = False
        portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                      for mu in mus]

        ## CALCULATE RISKS AND RETURNS FOR FRONTIER
        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

        ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        # CALCULATE THE OPTIMAL PORTFOLIO
        wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
        return pd.Series(list(wt), index=self.cov.index)

    def get_IVP(self, cov: pd.DataFrame = None) -> pd.Series:
        """
            Calculate and return the portfolio that is optimal in terms of the inverse variance portfolio (IVP).

            Args:
                cov (pandas DataFrame): optional covariance matrix, e.g. needed if IVP should be calculated for
                                        different clusters in HRP, defaults to IVP of whole training data
        """
        if cov is None:
            cov = self.cov
        # Compute the inverse-variance portfolio
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return pd.Series(ivp, index=cov.index)

    def get_HRP(self) -> pd.Series:
        """
            Calculate and return the portfolio that is optimal in terms of the hierarchical risk parity (HRP).
        """
        # Construct a hierarchical portfolio
        sortIx = self.get_quasi_diag()
        sortIx = self.corr.index[sortIx].tolist()
        hrp = self.get_rec_bipart(sortIx)
        return hrp.sort_index()

    def get_all_portfolios(self) -> pd.DataFrame:
        """
            Calculate and return all three currently available portfolio strategies:
                - Hierarchical Risk Parity (HRP)
                - Inverse Variance Portfolio (IVP)
                - Critical Line Algorithm (CLA)
        """
        cla = self.get_CLA()
        ivp = self.get_IVP()
        hrp = self.get_HRP()
        portfolios = pd.DataFrame({'CLA': cla, 'IVP': ivp, 'HRP': hrp})
        return portfolios

    def get_results(self, portfolios: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
        """
            Calculate and return the in sample and out of sample results for given portfolios.

            Args:
                portfolios (pandas DataFrame): dataframe of one or more portfolios
        """
        if type(portfolios) != pd.DataFrame:
            raise ValueError(
                f'HierarchRiskParity Class - method get_results - Parameter portfolios has to be of type'
                f'pandas.DataFrame, but got type {type(portfolios)}')

        in_sample_result = pd.DataFrame(np.dot(self.returns_train, np.array(portfolios)),
                                        columns=portfolios.columns, index=self.returns_train.index)
        out_of_sample_result = pd.DataFrame(np.dot(self.returns_test, np.array(portfolios)),
                                            columns=portfolios.columns, index=self.returns_test.index)
        return [in_sample_result, out_of_sample_result]

    def get_stdev_and_sharpe_ratio(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """
            Calculate and return the standard deviation and sharpe ratio for given results.

            Args:
                result_df (pandas DataFrame): dataframe of results of one or more portfolio applications
        """
        if type(result_df) != pd.DataFrame:
            raise ValueError(
                f'HierarchRiskParity Class - method show_result_plot - Parameter result_df has to be of type'
                f'pandas.DataFrame, but got type {type(result_df)}')
        stddev = result_df.std() * np.sqrt(252)
        sharp_ratio = (result_df.mean() * np.sqrt(252)) / result_df.std()
        return pd.DataFrame(dict(stdev=stddev, sharp_ratio=sharp_ratio))

    def hrp_MC(self, num_iters=1e4, shift_count=5, rebal=22):
        """
            Todo: The following code was taken from Marcos Lopez book and is not yet working fully, it still needs
             some work but the ground work of code is already implemented below.
            Calculate Monte Carlo experiment for hierarchical risk parity (HRP) to see portfolio shifts of HRP.

            Args (optional, defaults available):
                num_iters (int): Number of iterations for Monte Carlo
                shift_count (int): Number of portfolio shifts during the Monte Carlo iterations
                rebal (int): Number of portfolio rebalances during the Monte Carlo iterations
        """
        s_length = int(self.data_train.shape[0] / shift_count)
        methods = [self.get_IVP, self.get_HRP, self.get_CLA]
        stats, num_iter = {i.__name__: pd.Series() for i in methods}, 0
        pointers = range(s_length, self.data_train.shape[1], rebal)

        while num_iter < num_iters:
            # 1) Prepare data for one experiment
            x = self.data_train.iloc[num_iter:num_iter + s_length]
            r = {i.__name__: pd.Series() for i in methods}

            # 2) Compute portfolios in-sample
            for pointer in pointers:
                x_ = x[pointer - s_length:pointer]
                cov_, corr_ = np.cov(x_, rowvar=0), np.corrcoef(x_, rowvar=0)
                self.cov = cov_
                self.corr = corr_
                # 3) Compute performance out-of-sample
                x_ = x[pointer:pointer + rebal]
                for func in methods:
                    w_ = func()  # callback
                    r_ = pd.Series(np.dot(x_, w_))
                    r[func.__name__] = r[func.__name__].append(r_)

            # 4) Evaluate and store results
            for func in methods:
                r_ = r[func.__name__].reset_index(drop=True)
                p_ = (1 + r_).cumprod()
                stats[func.__name__].loc[num_iter] = p_.iloc[-1] - 1  # terminal return

            num_iter += 1

        # 5) Report results
        stats = pd.DataFrame.from_dict(stats, orient='columns')
        stats.to_csv('stats.csv')

        df0, df1 = stats.std(), stats.var()
        result_df = pd.concat([df0, df1, df1 / df1['getHRP'] - 1], axis=1)

        return result_df
