import pandas as pd


class CSVLoader:
    """
    Utility class for loading CSV datasets into pandas DataFrames.
    """

    def __init__(self, file_path: str):
        # Store dataset location for loading
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        # Read CSV file into memory as a pandas DataFrame
        df = pd.read_csv(self.file_path)

        return df