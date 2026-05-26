"""
data_loader.py
--------------
Responsible for loading the gas monitoring dataset from the SQLite
database specified in the pipeline configuration.

Author: [Your Name]
"""

import sqlite3
import pandas as pd


def load_data(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Load data from a SQLite database table into a Pandas DataFrame.

    Parameters
    ----------
    db_path : str
        Relative or absolute path to the .db file.
    table_name : str
        Name of the table to query.

    Returns
    -------
    pd.DataFrame
        Raw dataset as loaded from the database.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist at the given path.
    ValueError
        If the specified table does not exist in the database.
    """
    try:
        conn = sqlite3.connect(db_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not open database at '{db_path}'. "
            f"Ensure the file exists and is a valid SQLite database."
        ) from exc

    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception as exc:
        raise ValueError(
            f"Table '{table_name}' could not be found or queried in '{db_path}'."
        ) from exc
    finally:
        conn.close()

    print(f"[data_loader] Loaded {len(df):,} rows × {df.shape[1]} columns "
          f"from '{db_path}' → table '{table_name}'")
    return df
