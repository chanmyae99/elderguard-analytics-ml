"""
logistic_regression_model.py
----------------------------
Implements Logistic Regression as the baseline classification model.

Logistic Regression is used as an interpretable linear baseline for
comparison against tree-based models.
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
    Logistic Regression model wrapper.

    Uses scaled features from DataService.
    """

    def __init__(self):
        super().__init__("Logistic Regression")
        self.cv_scores = None

    def build_model(self):
        """
        Build Logistic Regression model using config parameters.
        """
        return LogisticRegression(
            C=LR_PARAMS["C"],
            max_iter=LR_PARAMS["max_iter"],
            solver=LR_PARAMS["solver"],
            random_state=RANDOM_STATE,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv_folds: int = CV_FOLDS,
    ):
        """
        Train Logistic Regression with stratified cross-validation.
        """

        self.model = self.build_model()

        print(f"[logistic_regression] Running {cv_folds}-fold CV...")

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
            "[logistic_regression] CV Accuracy: "
            f"{self.cv_scores.mean():.4f} ± {self.cv_scores.std():.4f}"
        )

        self.model.fit(X_train, y_train)

        print("[logistic_regression] Training complete.")

        return self.model