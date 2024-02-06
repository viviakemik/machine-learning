import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.metrics import mutual_info_score


class DistanceMetrics:
    def __init__(self, data, col=None):
        print(data)
        if data is not None:
            if col is None:
                col = data.columns
            self.data = pd.DataFrame(data, columns=col)
            self.filtered_data = self.data[col]
        else:
            self.data = None
            self.filtered_data = None
        self.col = col

    def pearson_cor(self):
        # Calculate Pearson correlation matrix
        correlation_matrix = self.filtered_data.corr()
        pearson_coefficient = correlation_matrix.loc[self.col]
        return correlation_matrix, pearson_coefficient

    def euclidean_dist(self):
        euc_dist = euclidean_distances(self.filtered_data.T)
        euc_dist_df = pd.DataFrame(euc_dist, index=self.col, columns=self.col)
        return euc_dist_df

    def corr_based_metric(self, corr: pd.DataFrame) -> pd.DataFrame:
        T = len(corr.iloc[:, 0])
        result = np.sqrt(2 * T * (1 - np.abs(corr)))
        return result

    def histogram2d(self, x, y, bins=None):
        if bins is None:
            bins = self.num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
        histogram, *_ = np.histogram2d(x, y, bins=bins)
        return histogram

    def marginal(self, x, bins=None):
        """
        Marginal entropy H[X] = -sum(Ni/N * log(Ni/N))
        :param x: input data
        :type x: array_like
        :param bins: the number of equal-width bins in the given range (10,
        by default)
        :type bins: int, optional
        :return: entropy is calculated as ``S = -sum(pk * log(pk), axis=axis)``.
        :rtype: float
        """
        if bins is None:
            bins = self.num_bins(x.shape[0])
        histogram, *_ = np.histogram(x, bins=bins)
        return ss.entropy(histogram)

    def joint(self, x, y, bins=None):
        """
        Joint entropy H[X,Y] = H[X] + H[Y] - I[X,Y]
        :param x: X observations
        :type x: array_like
        :param y: Y observations
        :type y: array_like
        :param bins: the number of equal-width bins in the given range (10,
        by default)
        :type bins: int, optional
        :return: Joint entropy
        :rtype: float
        """

        return (
            self.marginal(x, bins=bins)
            + self.marginal(y, bins=bins)
            - self.mutual_info(x, y, bins=bins)
        )

    def mutual_info(self, x, y, bins=None, norm=False):
        """
        Mutual Information : The informational gain in X that results from
        knowing the value of Y
        :param x: X observations
        :type x: array_like
        :param y: Y observations
        :type y: array_like
        :param bins: the number of equal-width bins in the given range (10,
        by default)
        :type bins: int, optional
        :param norm: Parameter to get the normalized version of the measure or
        not (False, by default)
        :type norm: bool, optional
        :return: Mutual Information I[X,Y] = H[X] - H[X|Y]
        :rtype:
        """
        corr = np.corrcoef(x, y)[0, 1]
        if corr == 1:
            bins = 0
        if bins is None:
            bins = self.num_bins(x.shape[0], corr=corr)

        mi = mutual_info_score(
            None, None, contingency=self.histogram2d(x, y, bins=bins)
        )

        if norm:
            return mi / min(self.marginal(x, bins=bins), self.marginal(y, bins=bins))
        return mi

    def conditional(self, x, y, bins=None):
        """
        Conditional entropY H(X|Y) = H(X,Y) - H(Y)
        :param x: X observations
        :type x: array_like
        :param y: Y observations
        :type y: array_like
        :param bins: the number of equal-width bins in the given range (None,
        by default)
        :type bins: int, optional
        :return: conditional entropy
        :rtype: float
        """
        joint = self.joint(x, y, bins=bins)
        marginal = self.marginal(y, bins=bins)
        return self.joint(x, y, bins=bins) - self.marginal(y, bins=bins)

    def variation_info(self, x, y, bins=None, norm=False):
        """
        Variation info VI(X,Y) = H(X|Y) + H(Y|X) = H(X) + H(Y) - 2 I(X,Y)
        :param x: X observations
        :type x: array_like
        :param y: Y observations
        :type y: array_like
        :param bins: the number of equal-width bins in the given range (10,
        by default)
        :type bins: int, optional
        :param norm: Parameter to get the normalized version of the measure or
        not (False, by default)
        :type norm: bool, optional
        :return: variation info
        :rtype: float
        """
        if bins is None:
            bins = self.num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])

        i_xy = self.mutual_info(x, y, bins=bins)
        h_x = self.marginal(x, bins=bins)
        h_y = self.marginal(y, bins=bins)

        v_xy = h_x + h_y - 2 * i_xy
        if norm:
            h_xy = h_x + h_y - i_xy
            return v_xy / h_xy
        return v_xy

    def num_bins(self, n_obs, corr=None):
        """
        Optimal number of bins for discretization
        :param n_obs: number of observations
        :type n_obs: int
        :param corr: Correlation between X, Y
        :type corr: float
        :return: Optimal number of bins
        :rtype: int
        """
        if corr is None:
            z = (8 + 324 * n_obs + 12 * (36 * n_obs + 729 * n_obs**2) ** 0.5) ** (
                1 / 3.0
            )

            b = round(z / 6.0 + 2.0 / (3 * z) + 1.0 / 3)
        else:
            try:
                b = round(
                    2 ** (-0.5) * (1 + (1 + 24 * n_obs / (1 - corr**2)) ** 0.5) ** 0.5
                )

            except (ZeroDivisionError, OverflowError):
                raise ValueError(
                    f"To use the optimal bining for joint entropy, "
                    f"the correlation should not be equal to 1 or -1. "
                    f"The correlation given is equal to {corr}"
                )
                return self.num_bins(n_obs)

        return int(b)
