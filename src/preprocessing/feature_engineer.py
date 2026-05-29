"""
feature_engineer.py
-------------------
Transforms the loaded dataset into a machine-learning-ready format.

Steps performed:
  1. Drop non-predictive columns (Session ID)
  2. Separate features (X) and target (y)
  3. Encode target with LabelEncoder
  4. One-hot encode categorical features
  5. Apply StandardScaler for Logistic Regression only

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.ingestion.csv_loader import CSVLoader
from src.utils.config import (
    PROCESSED_DATA_PATH,
    TARGET_COL,
    DROP_COLS,
    CATEGORICAL_COLS,
)


class FeatureEngineer:
    """
    Transforms a cleaned DataFrame into model-ready features and labels.

    Attributes
    ----------
    apply_scaling : bool
        True for Logistic Regression (StandardScaler applied).
        False for Random Forest and XGBoost.
    label_encoder : LabelEncoder
        Fitted encoder for the target column.
    scaler : StandardScaler or None
        Fitted scaler — only when apply_scaling=True.
    feature_columns : list
        Column names after one-hot encoding. Used to align test data
        to the exact same columns as training data.
    """

    def __init__(self, apply_scaling: bool = False):
        self.apply_scaling   = apply_scaling
        self.label_encoder   = LabelEncoder()
        self.scaler          = StandardScaler() if apply_scaling else None
        self.feature_columns = None

    def fit_transform(self, df: pd.DataFrame):
        """
        Fit on training data and transform it.

        Parameters
        ----------
        df : pd.DataFrame
            Training partition.

        Returns
        -------
        X : pd.DataFrame
            Encoded feature matrix.
        y : np.ndarray
            Integer-encoded target labels.
        """
        df = df.copy()

        # Step 1 — Drop non-predictive columns
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # Step 2 — Separate features and target
        y_raw = df.pop(TARGET_COL)

        # Step 3 — Encode target
        y = self.label_encoder.fit_transform(y_raw.astype(str))
        print(f"[feature_engineer] Target encoding: "
              f"{ {c: i for i, c in enumerate(self.label_encoder.classes_)} }")

        # Step 4 — One-hot encode categorical features
        # Reason: categories have no numerical order — OHE prevents
        # false mathematical relationships between categories
        X = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
        self.feature_columns = X.columns.tolist()
        print(f"[feature_engineer] Features after OHE: "
              f"{len(self.feature_columns)}")

        # Step 5 — Scale (Logistic Regression only)
        if self.apply_scaling:
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=self.feature_columns
            )
            print("[feature_engineer] StandardScaler applied")

        return X, y

    def transform(self, df: pd.DataFrame):
        """
        Apply fitted transformations to test data.

        Parameters
        ----------
        df : pd.DataFrame
            Test partition.

        Returns
        -------
        X : pd.DataFrame
            Encoded feature matrix aligned to training columns.
        y : np.ndarray
            Integer-encoded target labels.
        """
        df = df.copy()
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        y_raw = df.pop(TARGET_COL)
        y = self.label_encoder.transform(y_raw.astype(str))

        X = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

        # Align to training columns — handles any unseen categories
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        if self.apply_scaling:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=self.feature_columns
            )

        return X, y

    @property
    def class_names(self):
        """Return class names in encoded order."""
        return list(self.label_encoder.classes_)


# ── Standalone usage (matches existing file pattern) ──────────
if __name__ == "__main__":
    loader = CSVLoader(PROCESSED_DATA_PATH)
    df = loader.load()
    print(df.head())
