import sqlite3
import pandas as pd


class SQLiteLoader:
    """
    Utility class for loading data from SQLite database tables
    into pandas DataFrames.
    """
    def __init__(self, db_path: str):
        
        # Store database file path for future connections
        self.db_path = db_path

    def load_table(self, table_name: str) -> pd.DataFrame:

        connection = sqlite3.connect(self.db_path)

        df = pd.read_sql_query(
            f"SELECT * FROM {table_name}",
            connection
        )

        connection.close()

        return df