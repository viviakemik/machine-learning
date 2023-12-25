import pandas as pd

class ParquetLoader:
    def __init__(self, parquet_file_path):
        self.parquet_file_path = parquet_file_path

    def load_parquet(self):
        try:
            # Load the Parquet data into a DataFrame
            df = pd.read_parquet(self.parquet_file_path)

            return df
        except Exception as e:
            return None
