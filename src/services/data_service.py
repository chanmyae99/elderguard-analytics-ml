"""
data_service.py
---------------
Orchestrates data loading and preprocessing.

Acts as the entry point of the data pipeline — loads the CSV,
runs feature engineering, splits into train/test, and optionally
handles class imbalance.

Author: [Your Name]
"""

import pandas as pd

from src.ingestion.csv_loader import CSVLoader
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.imbalance_handler import ImbalanceHandler
from src.utils.config import (
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
)


class DataService:
    """
    Loads and prepares data for model training and evaluation.

    Attributes
    ----------
    fe_tree : FeatureEngineer
        Feature engineer for tree-based models (no scaling).
    fe_lr : FeatureEngineer
        Feature engineer for Logistic Regression (with scaling).
    class_names : list
        Human-readable class labels in encoded order.
    """

    def __init__(self, apply_imbalance_handling: bool = False):
        """
        Args:
            apply_imbalance_handling: Set True to apply SMOTE on
                training data. Default False — train baseline first.
        """
        self.apply_imbalance_handling = apply_imbalance_handling
        self.fe_tree    = FeatureEngineer(apply_scaling=False)
        self.fe_lr      = FeatureEngineer(apply_scaling=True)
        self.class_names = None

    def prepare(self):
        """
        Load data and return train/test splits for all models.

        Returns
        -------
        dict with keys:
            X_train_tree, X_test_tree  — for Random Forest and XGBoost
            X_train_lr, X_test_lr      — for Logistic Regression (scaled)
            y_train, y_test            — shared encoded labels
        """
        # Load CSV
        loader = CSVLoader(PROCESSED_DATA_PATH)
        df = loader.load()
        print(f"[data_service] Loaded {len(df):,} rows × "
              f"{df.shape[1]} columns")

        # Split DataFrame first — fit encoders on train only
        splitter = DataSplitter()
        from src.utils.config import TARGET_COL
        train_df, test_df = self._split_df(df, splitter, TARGET_COL)

        # Feature engineering — tree models
        X_train_tree, y_train = self.fe_tree.fit_transform(train_df)
        X_test_tree,  y_test  = self.fe_tree.transform(test_df)

        # Feature engineering — logistic regression (scaled)
        X_train_lr, _ = self.fe_lr.fit_transform(train_df)
        X_test_lr,  _ = self.fe_lr.transform(test_df)

        self.class_names = self.fe_tree.class_names
        print(f"[data_service] Classes: {self.class_names}")

        # Optional SMOTE
        imb = ImbalanceHandler(
            enabled=self.apply_imbalance_handling,
            random_state=RANDOM_STATE
        )
        X_train_tree, y_train = imb.handle(X_train_tree, y_train)
        X_train_lr,   _       = imb.handle(X_train_lr, y_train)

        return {
            "X_train_tree": X_train_tree,
            "X_test_tree" : X_test_tree,
            "X_train_lr"  : X_train_lr,
            "X_test_lr"   : X_test_lr,
            "y_train"     : y_train,
            "y_test"      : y_test,
        }

    def _split_df(self, df: pd.DataFrame,
                  splitter: DataSplitter,
                  target_col: str):
        """Split raw DataFrame into train/test before encoding."""
        from sklearn.model_selection import train_test_split
        from src.utils.config import TEST_SIZE
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            stratify=df[target_col],
            random_state=RANDOM_STATE,
        )
        print(f"[data_service] Train: {len(train_df):,} | "
              f"Test: {len(test_df):,}")
        return train_df, test_df
