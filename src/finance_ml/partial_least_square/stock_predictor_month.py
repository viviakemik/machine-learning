import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from data_loader import ParquetLoader

class StockPredictor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def run_analysis(self):
       
        # Define the path to the Parquet file
        parquet_file_path = 'data/fixed_income/IGIB_2020-04-07_2022-04-06.parquet'

        # Create an instance of ParquetLoader
        loader = ParquetLoader(parquet_file_path)

        # Load the Parquet data into a DataFrame
        df = loader.load_parquet()

        if df is not None:
            # Display the first few rows of the DataFrame
            print("----------------load DATAFRAME------")
            print(df.head() )
            data=df
             
                    # Convert 'DATE' column to datetime format
            data['DATE'] = pd.to_datetime(data['DATE'])

            # Set 'DATE' column as the index
            data.set_index('DATE', inplace=True)

            # Group the data by month and calculate the mean for each month
            data = data.resample('M').mean()

            # Display the first few rows of the monthly data
            # print("----------------Monthly Data------")
            # print(data.head())
            data.reset_index(inplace=True)
            data['Next_Close'] = data['CLOSE'].shift(-1)  
            data['Day'] = data['DATE'].dt.day
            data['Month'] = data['DATE'].dt.month
            data['Year'] = data['DATE'].dt.year
            data = data[['Day', 'Month', 'Year', 'Next_Close']]
            data.dropna(inplace=True)

            # Separate features and target
            X = data[['Day', 'Month', 'Year']]
            y = data['Next_Close']

            print(X)
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
            y_pred = model.predict(X_test_scaled)

            # Display metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print("Mean Squared Error:", mse)
            print("R-squared:", r2)
            # Create a date range based on the 'Day', 'Month', and 'Year' columns
            data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])

            # Set 'Date' column as the index
            dates = data.index[-len(y_test):]

            # Visualization
            # Creating a dummy date range from 2020 to 2022 named datesx
            datesx = pd.date_range(start='2020-01-01', end='2022-3-11', periods=len(y_test))

            plt.figure(figsize=(12, 6))
            plt.plot(datesx, y_test, label='Actual Next Close Price', color='blue')
            plt.plot(datesx, y_pred, label='Predicted Next Close Price', color='red', linestyle='dashed')
            plt.title(f'{self.ticker} Stock Prediction - Actual vs Predicted Next Closing Prices')
            plt.xlabel('DatesX')
            plt.ylabel('Next Close Price')
            plt.legend()

            plt.show()

        else:
            print("Failed to load Parquet data.")
       