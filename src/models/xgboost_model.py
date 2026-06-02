"""
xgboost_model.py
----------------
Implements XGBoost as a gradient boosting classifier.

XGBoost is designed to capture complex non-linear relationships
and is expected to provide strong predictive performance.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.base_model import BaseModel
from src.utils.config import (
    XGB_PARAMS,
    CV_FOLDS,
    RANDOM_STATE,
)


class XGBoostModel(BaseModel):
    """
    XGBoost model wrapper.

    Uses unscaled features from DataService.
    """

    def __init__(self):
        super().__init__("XGBoost")
        self.cv_scores = None

    def build_model(self):
        """
        Build XGBoost model using config parameters.
        """
        return XGBClassifier(
            n_estimators=XGB_PARAMS["n_estimators"],
            learning_rate=XGB_PARAMS["learning_rate"],
            max_depth=XGB_PARAMS["max_depth"],
            subsample=XGB_PARAMS["subsample"],
            colsample_bytree=XGB_PARAMS["colsample_bytree"],
            random_state=XGB_PARAMS["random_state"],
            objective="multi:softprob",
            eval_metric="mlogloss",
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv_folds: int = CV_FOLDS,
    ):
        """
        Train XGBoost with stratified cross-validation.
        """

        self.model = self.build_model()

        print(f"[xgboost] Running {cv_folds}-fold CV...")

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        self.cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        print(
            "[xgboost] CV Accuracy: "
            f"{self.cv_scores.mean():.4f} ± {self.cv_scores.std():.4f}"
        )

        self.model.fit(X_train, y_train)

        print("[xgboost] Training complete.")

        return self.model