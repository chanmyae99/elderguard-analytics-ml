"""
data_service.py
---------------
Coordinates data loading, feature engineering, splitting, and optional
class imbalance handling for the ML pipeline.
"""

import pandas as pd

from src.ingestion.csv_loader import CSVLoader
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.imbalance_handler import ImbalanceHandler
from src.utils.config import (
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COL,
)


class DataService:
    """
    Prepares model-ready train and test datasets.

    Logistic Regression uses scaled features.
    Tree-based models use unscaled features.
    """

    def __init__(self, apply_imbalance_handling: bool = False):
        self.apply_imbalance_handling = apply_imbalance_handling

        # Tree-based models do not require feature scaling
        self.tree_feature_engineer = FeatureEngineer(apply_scaling=False)

        # Logistic Regression requires scaled features for stable optimization
        self.lr_feature_engineer = FeatureEngineer(apply_scaling=True)


        self.class_names = None

    def prepare(self):
        """
        Load data, split into train/test, apply feature engineering,
        and optionally apply SMOTE to training data only.

        Returns:
            dict: Prepared datasets for Logistic Regression and tree models.
        """

        df = self._load_data()

        # Perform train-test split before feature engineering to prevent data leakage
        train_df, test_df = self._split_dataframe(df)

        # Generate unscaled features for tree-based models
        X_train_tree, y_train = self.tree_feature_engineer.fit_transform(
            train_df
        )
        X_test_tree, y_test = self.tree_feature_engineer.transform(
            test_df
        )

        # Generate scaled features for Logistic Regression
        X_train_lr, _ = self.lr_feature_engineer.fit_transform(
            train_df
        )
        X_test_lr, _ = self.lr_feature_engineer.transform(
            test_df
        )

        # Store class labels for reporting and visualization
        self.class_names = self.tree_feature_engineer.class_names

        if self.apply_imbalance_handling:

            # Apply SMOTE only to the training data to avoid contaminating
            # the test set with synthetic samples
            X_train_tree, X_train_lr, y_train = self._apply_smote(
                X_train_tree,
                X_train_lr,
                y_train
            )

        return {
            "X_train_tree": X_train_tree,
            "X_test_tree": X_test_tree,
            "X_train_lr": X_train_lr,
            "X_test_lr": X_test_lr,
            "y_train": y_train,
            "y_test": y_test,
            "class_names": self.class_names,
        }

    def _load_data(self) -> pd.DataFrame:

        # Load the cleaned dataset produced by the preprocessing pipeline
        loader = CSVLoader(PROCESSED_DATA_PATH)
        df = loader.load()

        print(
            f"[data_service] Loaded dataset: "
            f"{df.shape[0]:,} rows × {df.shape[1]} columns"
        )

        return df

    def _split_dataframe(self, df: pd.DataFrame):
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,

            # Preserve class distribution across train and test sets
            stratify=df[TARGET_COL],
        )

        print(
            f"[data_service] Train rows: {len(train_df):,} | "
            f"Test rows: {len(test_df):,}"
        )

        return train_df, test_df

    def _apply_smote(self, X_train_tree, X_train_lr, y_train):
        """
        Apply SMOTE to training data only.

        Important:
        The same y_train must be used for both feature versions.
        """

        # Initialize SMOTE handler for minority class oversampling
        handler = ImbalanceHandler(
            enabled=True,
            random_state=RANDOM_STATE
        )
        
        # Resample tree-model features and target labels
        X_train_tree_resampled, y_train_resampled = handler.handle(
            X_train_tree,
            y_train
        )

        # Resample Logistic Regression features using the same target labels
        # to keep both feature representations aligned
        X_train_lr_resampled, _ = handler.handle(
            X_train_lr,
            y_train
        )

        return (
            X_train_tree_resampled,
            X_train_lr_resampled,
            y_train_resampled,
        )


