from src.ingestion.csv_loader import CSVLoader

from src.utils.config import (
    PROCESSED_DATA_PATH
)

loader = CSVLoader(
    PROCESSED_DATA_PATH
)

df = loader.load()
print(df.head())