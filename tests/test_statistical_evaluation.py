import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.finance_ml.statistical_analysis.statistical_analysis import StatisticalEvaluation

# Mock data setup
X_data = pd.DataFrame({
    'DATE': pd.date_range(start='2020-01-01', periods=5),
    'IGIB_CLOSE': np.linspace(100, 105, 5),
    'Day': range(1, 6),
    'Month': [1] * 5,
    'Year': [2020] * 5,
    'OPEN': np.linspace(100, 104, 5),
    'HIGHT': np.linspace(101, 105, 5),
    'LOW': np.linspace(99, 103, 5),
    'CLOSE': np.linspace(100, 104, 5),
    'VOLUME': np.linspace(1000, 1004, 5),
    'VW': np.linspace(100, 104, 5),
    'TRANSACTIONS': np.linspace(10, 14, 5),
}).set_index('DATE')

y_data = pd.Series(np.linspace(105, 110, 5), index=pd.date_range(start='2020-01-01', periods=5))

# Mock model and its return values
mock_model = MagicMock()
mock_model.fit = MagicMock()
mock_model.predict = MagicMock(return_value=np.array([105, 106, 107, 108, 109]))


class TestStatisticalEvaluation(unittest.TestCase):
    def setUp(self):
        self.mock_data_loader = MagicMock()
        self.mock_data_loader.load_dataset.return_value = pd.DataFrame({
            'DATE': pd.date_range(start='2020-01-01', periods=5),
            'IGIB_CLOSE': np.linspace(100, 105, 5),
            'Day': range(1, 6),
            'Month': [1] * 5,
            'Year': [2020] * 5,
            'OPEN': np.linspace(100, 104, 5),
            'HIGHT': np.linspace(101, 105, 5),
            'LOW': np.linspace(99, 103, 5),
            'CLOSE': np.linspace(100, 104, 5),
            'VOLUME': np.linspace(1000, 1004, 5),
            'VW': np.linspace(100, 104, 5),
            'TRANSACTIONS': np.linspace(10, 14, 5),
        }).set_index('DATE')

        self.evaluation = StatisticalEvaluation()
        self.evaluation.dataloader = self.mock_data_loader

    def test_calculate_statistics(self):
        returns = pd.Series(np.random.randn(100))
        stats = self.evaluation.calculate_statistics(returns)

        self.assertIsInstance(stats, dict)
        self.assertIn('Mean return', stats)
        self.assertIn('Minimum', stats)
        self.assertIn('Maximum', stats)
        self.assertIn('Standard deviation', stats)

    @patch('src.finance_ml.statistical_analysis.statistical_analysis.Strategy')
    def test_simulation(self, MockStrategy):
        MockStrategy.return_value.simulate.return_value = pd.DataFrame({
            'portfolio_value': np.random.rand(5) * 1000
        })

        modelData = pd.DataFrame({
            'DATE': pd.date_range(start='2020-01-01', periods=5),
            'CLOSE': np.random.rand(5) * 100,
            'PNClose': np.random.rand(5) * 100,
        })

        history_df, strategy_data = self.evaluation.simulation(modelData)

        self.assertTrue(history_df.empty)
        self.assertTrue(strategy_data.empty)

    def test_evaluate_Stats(self):
        data = pd.DataFrame({
            'quantity': np.random.rand(10),
            'close_price': np.random.rand(10) * 100,
            'commission': np.random.rand(10),
            'return_before_costs': np.random.randn(10),
            'return_after_costs': np.random.randn(10),
        })

        stats_before, stats_after = self.evaluation.evaluate_Stats(data)

        self.assertFalse(stats_before.empty)
        self.assertFalse(stats_after.empty)
        self.assertIn('Statistic', stats_before.columns)
        self.assertIn('Value', stats_before.columns)

    @patch('src.finance_ml.statistical_analysis.statistical_analysis.DataLoader')
    @patch('src.finance_ml.statistical_analysis.statistical_analysis.PLSRegression', return_value=mock_model)
    @patch('src.finance_ml.statistical_analysis.statistical_analysis.StandardScaler')
    @patch('src.finance_ml.statistical_analysis.statistical_analysis.train_test_split',
           return_value=(X_data, X_data, y_data, y_data))
    def test_plsRegression(self, mock_train_test_split, mock_StandardScaler, mock_PLSRegression,
                           mock_DataLoader):
        # Setup mock DataLoader to return mock data
        mock_data_loader_instance = mock_DataLoader.return_value
        mock_data_loader_instance.load_dataset.return_value = X_data

        # Instantiate StatisticalEvaluation
        evaluation = StatisticalEvaluation()

        # Call plsRegression
        result_df = evaluation.plsRegression('fake_filename.csv')

        # Assertions
        mock_DataLoader.assert_called_once_with(time_index_col='DATE',
                                                keep_cols=['VOLUME', 'OPEN', 'HIGHT', 'LOW', 'CLOSE', 'VW',
                                                           'TRANSACTIONS'])
        mock_StandardScaler.assert_called()
        mock_PLSRegression.assert_called_once()
        mock_model.fit.assert_called()
        mock_model.predict.assert_called()

        self.assertIsInstance(result_df, pd.DataFrame, "The result should be a pandas DataFrame")
        self.assertTrue('PNClose' in result_df.columns, "Expected 'PNClose' column in the result DataFrame")
        # Verify that the returned DataFrame includes the expected columns after aggregation
        expected_columns = ['DATE', 'OPEN', 'HIGHT', 'LOW', 'CLOSE', 'VOLUME', 'VW', 'TRANSACTIONS', 'PNClose']
        for col in expected_columns:
            self.assertIn(col, result_df.columns, f"Expected column '{col}' in the result DataFrame")

    @patch('src.finance_ml.statistical_analysis.statistical_analysis.DataLoader')
    @patch('src.finance_ml.statistical_analysis.statistical_analysis.RandomForestRegressor', return_value=mock_model)
    @patch('src.finance_ml.statistical_analysis.statistical_analysis.StandardScaler')
    @patch('src.finance_ml.statistical_analysis.statistical_analysis.train_test_split',
           return_value=(X_data, X_data, y_data, y_data))
    def test_RandonForestRegression(self, mock_train_test_split, mock_StandardScaler, mock_RandomForestRegressor,
                                    mock_DataLoader):
        # Setup mock DataLoader to return mock data
        mock_data_loader_instance = mock_DataLoader.return_value
        mock_data_loader_instance.load_dataset.return_value = X_data

        # Instantiate StatisticalEvaluation
        evaluation = StatisticalEvaluation()

        # Call plsRegression
        result_df = evaluation.randomForest('fake_filename.csv')

        # Assertions
        mock_DataLoader.assert_called_once_with(time_index_col='DATE',
                                                keep_cols=['VOLUME', 'OPEN', 'HIGHT', 'LOW', 'CLOSE', 'VW',
                                                           'TRANSACTIONS'])
        mock_StandardScaler.assert_called()
        mock_RandomForestRegressor.assert_called_once()
        mock_model.fit.assert_called()
        mock_model.predict.assert_called()

        self.assertIsInstance(result_df, pd.DataFrame, "The result should be a pandas DataFrame")
        self.assertTrue('PNClose' in result_df.columns, "Expected 'PNClose' column in the result DataFrame")
        # Verify that the returned DataFrame includes the expected columns after aggregation
        expected_columns = ['DATE', 'OPEN', 'HIGHT', 'LOW', 'CLOSE', 'VOLUME', 'VW', 'TRANSACTIONS', 'PNClose']
        for col in expected_columns:
            self.assertIn(col, result_df.columns, f"Expected column '{col}' in the result DataFrame")

    @patch('src.finance_ml.statistical_analysis.statistical_analysis.DataLoader')
    @patch('src.finance_ml.statistical_analysis.statistical_analysis.GradientBoostingRegressor',
           return_value=mock_model)
    @patch('src.finance_ml.statistical_analysis.statistical_analysis.StandardScaler')
    @patch('src.finance_ml.statistical_analysis.statistical_analysis.train_test_split',
           return_value=(X_data, X_data, y_data, y_data))
    def test_Gradientboosting(self, mock_train_test_split, mock_StandardScaler, mock_GradientBoostingRegressor,
                              mock_DataLoader):
        # Setup mock DataLoader to return mock data
        mock_data_loader_instance = mock_DataLoader.return_value
        mock_data_loader_instance.load_dataset.return_value = X_data

        # Instantiate StatisticalEvaluation
        evaluation = StatisticalEvaluation()

        # Call plsRegression
        result_df = evaluation.gradientBoosting('fake_filename.csv')

        # Assertions
        mock_DataLoader.assert_called_once_with(time_index_col='DATE',
                                                keep_cols=['VOLUME', 'OPEN', 'HIGHT', 'LOW', 'CLOSE', 'VW',
                                                           'TRANSACTIONS'])
        mock_StandardScaler.assert_called()
        mock_GradientBoostingRegressor.assert_called_once()
        mock_model.fit.assert_called()
        mock_model.predict.assert_called()

        self.assertIsInstance(result_df, pd.DataFrame, "The result should be a pandas DataFrame")
        self.assertTrue('PNClose' in result_df.columns, "Expected 'PNClose' column in the result DataFrame")
        # Verify that the returned DataFrame includes the expected columns after aggregation
        expected_columns = ['DATE', 'OPEN', 'HIGHT', 'LOW', 'CLOSE', 'VOLUME', 'VW', 'TRANSACTIONS', 'PNClose']
        for col in expected_columns:
            self.assertIn(col, result_df.columns, f"Expected column '{col}' in the result DataFrame")
