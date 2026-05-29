"""
logistic_regression_model.py
-----------------------------
Logistic Regression model for Activity Level prediction.

Inherits from BaseModel and imports hyperparameters from
src/utils/config.py (which reads config/config.yaml).

Role in the project
-------------------
Serves as the interpretable linear baseline. Its lower accuracy
compared to tree models confirms that activity boundaries in
sensor data are non-linear — itself a useful EDA finding.

Requires StandardScaler (apply_scaling=True in FeatureEngineer)
because Logistic Regression is sensitive to feature magnitude.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.base_model import BaseModel
from src.utils.config import (
    LR_PARAMS,
    CV_FOLDS,
    RANDOM_STATE,
)


class LogisticRegressionModel(BaseModel):
    """
    Wrapper around sklearn LogisticRegression.

    Attributes
    ----------
    model : LogisticRegression
        The underlying sklearn estimator.
    cv_scores : np.ndarray
        Accuracy scores from stratified cross-validation.
    """

    def __init__(self):
        self.model = LogisticRegression(
            C=LR_PARAMS["C"],
            max_iter=LR_PARAMS["max_iter"],
            solver=LR_PARAMS["solver"],
            random_state=RANDOM_STATE,
        )
        self.cv_scores = None

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray,
              cv_folds: int = CV_FOLDS, random_state: int = RANDOM_STATE):
        """
        Fit the model with stratified cross-validation.

        Args:
            X_train: Scaled training feature matrix (StandardScaler applied).
            y_train: Encoded training labels.
            cv_folds: Number of stratified folds.
            random_state: Seed for fold shuffle.
        """
        print(f"[logistic_regression] Running {cv_folds}-fold CV ...")
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        self.cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv, scoring="accuracy", n_jobs=-1
        )
        print(f"[logistic_regression] CV Accuracy: "
              f"{self.cv_scores.mean():.4f} ± {self.cv_scores.std():.4f}")
        self.model.fit(X_train, y_train)
        print("[logistic_regression] Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels for test data."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for test data."""
        return self.model.predict_proba(X)
