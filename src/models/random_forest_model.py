"""
random_forest_model.py
----------------------
Random Forest model for Activity Level prediction.

Inherits from BaseModel and imports hyperparameters from
src/utils/config.py (which reads config/config.yaml).

Role in the project
-------------------
Strong ensemble model. Handles non-linear sensor interactions
natively, requires no feature scaling, and uses
class_weight='balanced' to address class imbalance (High
Activity ~11%) without needing SMOTE.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.base_model import BaseModel
from src.utils.config import (
    RF_PARAMS,
    CV_FOLDS,
    RANDOM_STATE,
)


class RandomForestModel(BaseModel):
    """
    Wrapper around sklearn RandomForestClassifier.

    Attributes
    ----------
    model : RandomForestClassifier
        The underlying sklearn estimator.
    cv_scores : np.ndarray
        Accuracy scores from stratified cross-validation.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=RF_PARAMS["n_estimators"],
            max_depth=RF_PARAMS["max_depth"],
            min_samples_leaf=RF_PARAMS["min_samples_leaf"],
            max_features=RF_PARAMS["max_features"],
            class_weight=RF_PARAMS["class_weight"],
            n_jobs=RF_PARAMS["n_jobs"],
            random_state=RANDOM_STATE,
        )
        self.cv_scores = None

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray,
              cv_folds: int = CV_FOLDS, random_state: int = RANDOM_STATE):
        """
        Fit the model with stratified cross-validation.

        Args:
            X_train: Training feature matrix (no scaling needed).
            y_train: Encoded training labels.
            cv_folds: Number of stratified folds.
            random_state: Seed for fold shuffle.
        """
        print(f"[random_forest] Running {cv_folds}-fold CV ...")
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        self.cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv, scoring="accuracy", n_jobs=-1
        )
        print(f"[random_forest] CV Accuracy: "
              f"{self.cv_scores.mean():.4f} ± {self.cv_scores.std():.4f}")
        self.model.fit(X_train, y_train)
        print("[random_forest] Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels for test data."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for test data."""
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        """Return Gini-based feature importances from the fitted forest."""
        return self.model.feature_importances_
