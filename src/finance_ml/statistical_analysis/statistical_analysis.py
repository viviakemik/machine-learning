import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.finance_ml.data_preparation.data_preparation import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.finance_ml.backtests.strategy import Strategy, MarketAction
from scipy.stats import skew, kurtosis


class StatisticalEvaluation:
    """
    A class to load and preprocess stock data using DataLoader.
    """

    def __init__(self):
        """
        Initialize the StockDataLoader class.
        """
        self.dataloader = DataLoader(time_index_col='DATE',
                                     keep_cols=['VOLUME', 'OPEN', 'HIGHT', 'LOW', 'CLOSE', 'VW', 'TRANSACTIONS'])

    def calculate_statistics(self, returns):
        """
        Calculate various statistics for the returns.
        """
        statistics = {
            'Mean return': returns.mean(),
            'Minimum': returns.min(),
            'Quartile 1': returns.quantile(0.25),
            'Median': returns.median(),
            'Quartile 3': returns.quantile(0.75),
            'Maximum': returns.max(),
            'Standard deviation': returns.std(),
            'Skewness': skew(returns, nan_policy='omit'),
            'Kurtosis': kurtosis(returns, nan_policy='omit'),
            'Historical 1-percent VaR': returns.quantile(0.01),
            'Historical 1-percent CVaR': returns[returns <= returns.quantile(0.01)].mean(),
            'Historical 5-percent VaR': returns.quantile(0.05),
            'Historical 5-percent CVaR': returns[returns <= returns.quantile(0.05)].mean(),
            'Share with return > 0': (returns > 0).mean(),
        }
        # Calculate maximum drawdown and Calmar ratio
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = cumulative_returns / peak - 1
        statistics['Maximum drawdown'] = drawdown.min()
        annualized_return = returns.mean() * 252  # Assuming 252 trading days
        statistics['Calmar ratio'] = annualized_return / abs(statistics['Maximum drawdown'])

        return statistics

    def simulation(self, modelData):
        # Define the range of threshold values to test
        threshold_values = np.arange(0.001, 0.01, 0.001)

        # Initialize variables to store the best threshold and its corresponding profit
        best_threshold = None
        max_profit = 0

        # Iterate over threshold values
        for threshold in threshold_values:
            # Create a new instance of the strategy for each iteration
            strategy = Strategy(name="Simple Strategy", cash=1000, commission=0.05, min_positions=1)

            # Preprocess data based on the current threshold
            modelData['Buy_Signal'] = (modelData['PNClose'] / modelData['CLOSE']) - 1 > threshold
            modelData['Sell_Signal'] = (modelData['PNClose'] / modelData['CLOSE']) - 1 < -threshold

            strategy.load_data(modelData, date_column='DATE')

            # Define rules
            strategy.add_single_rule('Buy_Signal[N-1]==1', action=MarketAction.BUY, action_quantity='ALL')
            strategy.add_single_rule('Sell_Signal[N-1]==1', action=MarketAction.SELL, action_quantity='ALL')

            # Simulate the strategy
            strategy.simulate()

            # Get final portfolio value
            # Check if any trades were made
            if not strategy.history_df.empty:
                # Get final portfolio value
                final_portfolio_value = strategy.history_df['portfolio_value'].iloc[-1]
            else:
                # If no trades were made, use the starting cash value as the final portfolio value
                final_portfolio_value = 1000

            # Check if this is the best threshold so far
            if final_portfolio_value > max_profit:
                max_profit = final_portfolio_value
                best_threshold = threshold
        price_change_threshold = best_threshold

        strategy = Strategy(name="Simple Strategy", cash=1000, commission=0.05, min_positions=1)

        modelData['Buy_Signal'] = (modelData['PNClose'] / modelData['CLOSE']) - 1 > price_change_threshold
        modelData['Sell_Signal'] = (modelData['PNClose'] / modelData['CLOSE']) - 1 < -price_change_threshold
        strategy.load_data(modelData, date_column='DATE')

        # Create rules based on preprocessed data

        buy_rule = strategy.add_single_rule('Buy_Signal[N-1]==1', action=MarketAction.BUY, action_quantity='ALL')
        sell_rule = strategy.add_single_rule('Sell_Signal[N-1]==1', action=MarketAction.SELL, action_quantity='ALL')

        strategy.simulate()
        return strategy.history_df, strategy.data

    def evaluate_Stats(self, data):
        """
            Main function to process the financial data.
            """
        # Calculate returns
        """
        Calculate returns for trades in the dataset.
        Returns are calculated for each SELL action, both before and after transaction costs.
        """
        # Calculate returns
        data['return_before_costs'] = np.nan
        data['return_after_costs'] = np.nan
        for i in range(1, len(data), 2):  # Assuming every BUY is followed by a SELL
            buy_row = data.iloc[i - 1]
            sell_row = data.iloc[i]
            buy_amount = buy_row['quantity'] * buy_row['close_price']
            sell_amount = sell_row['quantity'] * sell_row['close_price']
            return_before_costs = (sell_amount - buy_amount) / buy_amount
            return_after_costs = ((sell_amount - sell_row['commission']) - (buy_amount + buy_row['commission'])) / (
                    buy_amount + buy_row['commission'])
            data.at[i, 'return_before_costs'] = return_before_costs
            data.at[i, 'return_after_costs'] = return_after_costs

        # Split returns into before and after transaction costs
        returns_before = data['return_before_costs'].dropna()
        returns_after = data['return_after_costs'].dropna()

        # Calculate statistics
        stats_before = self.calculate_statistics(returns_before)
        stats_after = self.calculate_statistics(returns_after)

        stats_before = pd.DataFrame(stats_before.items(), columns=['Statistic', 'Value'])
        stats_after = pd.DataFrame(stats_after.items(), columns=['Statistic', 'Value'])
        return stats_before, stats_after

    def plsRegression(self, filename):
        """
        Load data for the given stock ticker from the specified file.
        Preprocess data and then run the model to get a dataframe with
        predictions.

        Parameters:
            filename (str): File name containing the data.
        Returns:
            pd.DataFrame: dataset with predictions.
        """

        # ticker (str): Ticker symbol of the stock.
        ticker = 'IGIB'
        # n_records (int): Number of records to load, -1 to load all records.
        n_records = -1
        df = self.dataloader.load_dataset({ticker: filename}).iloc[:n_records]
        data = df.copy()

        # Fetch and prepare data
        data.reset_index(inplace=True)
        data['Next_Close'] = data['IGIB_CLOSE'].shift(-1)
        data['Day'] = data['DATE'].dt.day
        data['Month'] = data['DATE'].dt.month
        data['Year'] = data['DATE'].dt.year
        data.dropna(inplace=True)

        # Separate features and target
        X = data[['Day', 'Month', 'Year']]
        y = data['Next_Close']
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Standardize features for each feature to a mean of 0 and a standard deviation of 1
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train PLS Regression model
        model = PLSRegression(n_components=1)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred_pls = model.predict(X_test_scaled)

        # Display metrics
        mse = mean_squared_error(y_test, y_pred_pls)
        r2 = r2_score(y_test, y_pred_pls)
        print(y_pred_pls.shape)
        print("Mean Squared Error:", mse)
        print("R-squared:", r2)

        # Convert y_pred to a DataFrame and set the index to match X_test
        y_pred_df1 = pd.DataFrame(y_pred_pls, index=X_test.index, columns=['a_PNClose'])

        # Join/merge the predicted values with the original dataset
        data1 = data.join(y_pred_df1, how='left')

        """
        Make daily data from the preprocessed DataFrame.

        Parameters:
            df (pd.DataFrame): Preprocessed DataFrame.
        Returns:
            pd.DataFrame: DataFrame with daily aggregated data.
        """
        tdata = data1.copy()
        tdata.columns = [c.split('_')[-1] for c in tdata.columns]
        tdata = tdata.reset_index()

        # Make daily data
        tdata = tdata.set_index('DATE').resample('D')
        tdata = tdata.aggregate(
            {'OPEN': 'first', 'HIGHT': 'max', 'LOW': 'min', 'CLOSE': 'last', 'VOLUME': 'sum', 'VW': 'mean',
             'TRANSACTIONS': 'sum', 'PNClose': 'last'})
        tdata = tdata.dropna()
        tdata = tdata.reset_index()

        return tdata

    def randomForest(self, filename):
        """
         Load data for the given stock ticker from the specified file.
         Preprocess data and then run the model to get a dataframe with
         predictions.

         Parameters:
             filename (str): File name containing the data.
         Returns:
             pd.DataFrame: dataset with predictions.
         """
        ticker = 'IGIB'
        n_records = -1
        df = self.dataloader.load_dataset({ticker: filename}).iloc[:n_records]
        data = df.copy()

        # Fetch and prepare data
        data.reset_index(inplace=True)
        data['Next_Close'] = data['IGIB_CLOSE'].shift(-1)
        data['Day'] = data['DATE'].dt.day
        data['Month'] = data['DATE'].dt.month
        data['Year'] = data['DATE'].dt.year
        data.dropna(inplace=True)

        # Separate features and target
        X = data[['Day', 'Month', 'Year']]
        y = data['Next_Close']
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Standardize features for each feature to a mean of 0 and a standard deviation of 1
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest Regression model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred_rf = rf_model.predict(X_test_scaled)

        # Display metrics
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        print("Random Forest Mean Squared Error:", mse_rf)
        print("Random Forest R-squared:", r2_rf)

        # Now, you can use y_pred_rf for further analysis or visualization.

        # Convert y_pred to a DataFrame and set the index to match X_test
        y_pred_df2 = pd.DataFrame(y_pred_rf, index=X_test.index, columns=['a_PNClose'])

        # Join/merge the predicted values with the original dataset
        data2 = data.join(y_pred_df2, how='left')

        """
        Make daily data from the preprocessed DataFrame.

        Parameters:
            df (pd.DataFrame): Preprocessed DataFrame.
        Returns:
            pd.DataFrame: DataFrame with daily aggregated data.
        """
        tdata = data2.copy()
        tdata.columns = [c.split('_')[-1] for c in tdata.columns]
        tdata = tdata.reset_index()

        # Make daily data
        tdata = tdata.set_index('DATE').resample('D')
        tdata = tdata.aggregate(
            {'OPEN': 'first', 'HIGHT': 'max', 'LOW': 'min', 'CLOSE': 'last', 'VOLUME': 'sum', 'VW': 'mean',
             'TRANSACTIONS': 'sum', 'PNClose': 'last'})
        tdata = tdata.dropna()
        tdata = tdata.reset_index()

        return tdata

    def gradientBoosting(self, filename):
        """
         Load data for the given stock ticker from the specified file.
         Preprocess data and then run the model to get a dataframe with
         predictions.

         Parameters:
             filename (str): File name containing the data.
         Returns:
             pd.DataFrame: dataset with predictions.
         """
        ticker = 'IGIB'
        n_records = -1
        df = self.dataloader.load_dataset({ticker: filename}).iloc[:n_records]
        data = df.copy()

        # Fetch and prepare data
        data.reset_index(inplace=True)
        data['Next_Close'] = data['IGIB_CLOSE'].shift(-1)
        data['Day'] = data['DATE'].dt.day
        data['Month'] = data['DATE'].dt.month
        data['Year'] = data['DATE'].dt.year
        data.dropna(inplace=True)

        # Separate features and target
        X = data[['Day', 'Month', 'Year']]
        y = data['Next_Close']
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Standardize features for each feature to a mean of 0 and a standard deviation of 1
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Gradient Boosted Regression model
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
        gb_model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred_gb = gb_model.predict(X_test_scaled)

        # Display metrics
        mse_gb = mean_squared_error(y_test, y_pred_gb)
        r2_gb = r2_score(y_test, y_pred_gb)
        print("Gradient Boosted Regression Mean Squared Error:", mse_gb)
        print("Gradient Boosted Regression R-squared:", r2_gb)

        # Now, you can use y_pred_gb for further analysis or visualization.

        # Convert y_pred to a DataFrame and set the index to match X_test
        y_pred_df3 = pd.DataFrame(y_pred_gb, index=X_test.index, columns=['a_PNClose'])

        # Join/merge the predicted values with the original dataset
        data3 = data.join(y_pred_df3, how='left')

        # Convert y_pred to a DataFrame and set the index to match X_test
        y_pred_df3 = pd.DataFrame(y_pred_gb, index=X_test.index, columns=['a_PNClose'])

        # Join/merge the predicted values with the original dataset
        data3 = data.join(y_pred_df3, how='left')

        """
        Make daily data from the preprocessed DataFrame.

        Parameters:
            df (pd.DataFrame): Preprocessed DataFrame.
        Returns:
            pd.DataFrame: DataFrame with daily aggregated data.
        """
        tdata = data3.copy()
        tdata.columns = [c.split('_')[-1] for c in tdata.columns]
        tdata = tdata.reset_index()

        # Make daily data
        tdata = tdata.set_index('DATE').resample('D')
        tdata = tdata.aggregate(
            {'OPEN': 'first', 'HIGHT': 'max', 'LOW': 'min', 'CLOSE': 'last', 'VOLUME': 'sum', 'VW': 'mean',
             'TRANSACTIONS': 'sum', 'PNClose': 'last'})
        tdata = tdata.dropna()
        tdata = tdata.reset_index()

        return tdata
