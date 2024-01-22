import unittest
import pandas as pd
import numpy as np

from finance_ml.porfolio_optimization import PortfolioOptimization


class TestPortfolioOptimization(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        data_train = pd.DataFrame(np.random.randn(100, 5), index=dates,
                                  columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5'])
        data_test = pd.DataFrame(np.random.randn(50, 5), index=dates[-50:],
                                 columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5'])
        self.portfolio = PortfolioOptimization(data_train=data_train, data_test=data_test)

    def test_initialization(self):
        self.assertIsInstance(self.portfolio.data_train, pd.DataFrame)
        self.assertIsInstance(self.portfolio.data_test, pd.DataFrame)
        self.assertTrue(callable(self.portfolio.correl_dist))
        self.assertIsInstance(self.portfolio.drop_null, bool)
        self.assertIsInstance(self.portfolio.returns_train, pd.DataFrame)
        self.assertIsInstance(self.portfolio.returns_test, pd.DataFrame)
        self.assertIsInstance(self.portfolio.cov, pd.DataFrame)
        self.assertIsInstance(self.portfolio.corr, pd.DataFrame)

    def test_calc_linkage(self):
        linkage = self.portfolio.calc_linkage('single')
        self.assertIsInstance(linkage, np.ndarray)

    def test_get_quasi_diag(self):
        quasi_diag = self.portfolio.get_quasi_diag()
        self.assertIsInstance(quasi_diag, list)

    def test_get_cluster_var(self):
        sortIx = self.portfolio.get_quasi_diag()
        c_items = self.portfolio.corr.index[sortIx].tolist()
        cluster_var = self.portfolio.get_cluster_var(c_items)
        assert isinstance(cluster_var, np.ndarray) or isinstance(cluster_var,
                                                                 np.float64), "cluster_var is not an np.ndarray or np.float64"

    def test_get_rec_bipart(self):
        sort_ix = self.portfolio.get_quasi_diag()
        c_items = self.portfolio.corr.index[sort_ix].tolist()
        rec_bipart = self.portfolio.get_rec_bipart(c_items)
        self.assertIsInstance(rec_bipart, pd.Series)

    def test_get_CLA(self):
        cla_portfolio = self.portfolio.get_CLA()
        self.assertIsInstance(cla_portfolio, pd.Series)

    def test_get_IVP(self):
        ivp_portfolio = self.portfolio.get_IVP()
        self.assertIsInstance(ivp_portfolio, pd.Series)

    def test_get_HRP(self):
        hrp_portfolio = self.portfolio.get_HRP()
        self.assertIsInstance(hrp_portfolio, pd.Series)

    def test_get_all_portfolios(self):
        all_portfolios = self.portfolio.get_all_portfolios()
        self.assertIsInstance(all_portfolios, pd.DataFrame)
        self.assertEqual(all_portfolios.shape, (len(self.portfolio.cov.index), 3))

    def test_get_results(self):
        all_portfolios = self.portfolio.get_all_portfolios()
        results = self.portfolio.get_results(all_portfolios)
        self.assertIsInstance(results, list)
        self.assertIsInstance(results[0], pd.DataFrame)
        self.assertIsInstance(results[1], pd.DataFrame)

    def test_get_stdev_and_sharpe_ratio(self):
        all_portfolios = self.portfolio.get_all_portfolios()
        results = self.portfolio.get_results(all_portfolios)
        stdev_sharpe = self.portfolio.get_stdev_and_sharpe_ratio(results[0])
        self.assertIsInstance(stdev_sharpe, pd.DataFrame)
        self.assertEqual(stdev_sharpe.shape, (3, 2))


if __name__ == '__main__':
    unittest.main()
