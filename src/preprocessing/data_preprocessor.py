"""
data_preprocessor.py
--------------------
Preprocessing pipeline for the ElderGuard ML project.

Pipeline:
Raw SQLite Database
    ↓
Remove Duplicates
    ↓
Handle Missing Values
    ↓
Fix Temperature Values
    ↓
Validate Humidity
    ↓
Standardize Categorical Columns     
    ↓
Standardize Activity Labels
    ↓
Export Processed CSV
"""

import os
import pandas as pd

from src.ingestion.sqlite_loader import SQLiteLoader
from src.utils.config import (
    DB_PATH,
    DB_TABLE,
    PROCESSED_DATA_PATH,
)


class DataPreprocessor:

    def __init__(self):
        self.db_path = DB_PATH
        self.db_table = DB_TABLE
        self.output_path = PROCESSED_DATA_PATH

    def run(self) -> pd.DataFrame:

        df = self._load_raw_data()

        df = self._remove_duplicates(df)

        df = self._handle_missing_values(df)

        df = self._fix_temperature(df)

        df = self._validate_humidity(df)

        df = self._standardize_categorical_values(df)

        df = self._standardize_activity_labels(df)

        self._export_processed_data(df)

        return df

    def _load_raw_data(self) -> pd.DataFrame:

        loader = SQLiteLoader(self.db_path)

        df = loader.load_table(self.db_table)

        print(
            f"[data_preprocessor] Loaded raw data: "
            f"{df.shape[0]:,} rows × {df.shape[1]} columns"
        )

        return df

    def _remove_duplicates(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        print(
            f"[data_preprocessor] Removed duplicates: "
            f"{before - after:,}"
        )

        return df

    def _handle_missing_values(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        numerical_columns = [
            "Humidity",
            "MetalOxideSensor_Unit2",
            "CO_GasSensor"
        ]

        categorical_columns = [
            "Ambient Light Level"
        ]

        for col in numerical_columns:

            if col in df.columns:

                median_value = df[col].median()

                df[col] = df[col].fillna(
                    median_value
                )

        for col in categorical_columns:

            if col in df.columns:

                mode_value = df[col].mode()[0]

                df[col] = df[col].fillna(
                    mode_value
                )

        print(
            "[data_preprocessor] Missing values handled."
        )

        return df

    def _fix_temperature(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        if "Temperature" in df.columns:

            mask = df["Temperature"] > 100

            count = mask.sum()

            df.loc[
                mask,
                "Temperature"
            ] = (
                df.loc[
                    mask,
                    "Temperature"
                ] - 273.15
            )

            print(
                f"[data_preprocessor] "
                f"Converted {count:,} Kelvin values "
                f"to Celsius."
            )

        return df

    def _validate_humidity(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        before = len(df)

        df = df[
            (df["Humidity"] >= 0) &
            (df["Humidity"] <= 100)
        ]

        after = len(df)

        print(
            f"[data_preprocessor] Removed "
            f"{before - after:,} rows "
            f"with invalid humidity."
        )

        return df

    def _standardize_categorical_values(
        self,
        df: pd.DataFrame
        ) -> pd.DataFrame:

        if "HVAC Operation Mode" in df.columns:

            df["HVAC Operation Mode"] = (
                df["HVAC Operation Mode"]
                .astype(str)
                .str.strip()
                .str.lower()
                )

        print(
        "[data_preprocessor] Categorical values standardized."
        )

        return df

    def _standardize_activity_labels(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        activity_col = "Activity Level"

        if activity_col not in df.columns:
            return df

        df[activity_col] = (
            df[activity_col]
            .astype(str)
            .str.strip()
        )

        replacements = {
            "LowActivity": "Low",
            "Low Activity": "Low",
            "Low_Activity": "Low",

            "ModerateActivity": "Moderate",
            "Moderate Activity": "Moderate",
            "Moderate_Activity": "Moderate",

            "High Activity": "High",
            "High_Activity": "High",
        }

        df[activity_col] = (
            df[activity_col]
            .replace(replacements)
        )

        print(
            "[data_preprocessor] "
            "Activity labels standardized."
        )

        return df

    def _export_processed_data(
        self,
        df: pd.DataFrame
    ):

        os.makedirs(
            os.path.dirname(self.output_path),
            exist_ok=True
        )

        df.to_csv(
            self.output_path,
            index=False
        )

        print(
            f"[data_preprocessor] "
            f"Exported processed data to:\n"
            f"{self.output_path}"
        )


if __name__ == "__main__":

    processor = DataPreprocessor()

    processed_df = processor.run()

    print("\nProcessed Dataset Preview")
    print(processed_df.head())
    print(processed_df.shape)