"""
xgboost_model.py
----------------
XGBoost (Gradient Boosting) model for Activity Level prediction.

Inherits from BaseModel and imports hyperparameters from
src/utils/config.py (which reads config/config.yaml).

Role in the project
-------------------
Best-performing model on this dataset (~68% accuracy). Sequential
boosting corrects residual errors from prior trees, helping
distinguish the fuzzy Low/Moderate activity boundary in sensor
data. Outperforms Random Forest on accuracy.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.base_model import BaseModel
from src.utils.config import (
    XGB_PARAMS,
    CV_FOLDS,
    RANDOM_STATE,
)


class XGBoostModel(BaseModel):
    """
    Wrapper around xgboost.XGBClassifier.

    Attributes
    ----------
    model : XGBClassifier
        The underlying XGBoost estimator.
    cv_scores : np.ndarray
        Accuracy scores from stratified cross-validation.
    """

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=XGB_PARAMS["n_estimators"],
            max_depth=XGB_PARAMS["max_depth"],
            learning_rate=XGB_PARAMS["learning_rate"],
            subsample=XGB_PARAMS["subsample"],
            colsample_bytree=XGB_PARAMS["colsample_bytree"],
            n_jobs=XGB_PARAMS["n_jobs"],
            eval_metric="mlogloss",
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
        print(f"[xgboost] Running {cv_folds}-fold CV ...")
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        self.cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv, scoring="accuracy", n_jobs=-1
        )
        print(f"[xgboost] CV Accuracy: "
              f"{self.cv_scores.mean():.4f} ± {self.cv_scores.std():.4f}")
        self.model.fit(X_train, y_train)
        print("[xgboost] Training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels for test data."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for test data."""
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        """Return gain-based feature importances from the fitted model."""
        return self.model.feature_importances_
