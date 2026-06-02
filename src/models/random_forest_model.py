"""
random_forest_model.py
----------------------
Implements Random Forest as a tree-based ensemble classifier.

Random Forest is robust to non-linear relationships and provides
feature importance analysis for activity level prediction.
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
    Random Forest model wrapper.

    Uses unscaled features from DataService.
    """

    def __init__(self):
        super().__init__("Random Forest")
        self.cv_scores = None

    def build_model(self):
        """
        Build Random Forest model using config parameters.
        """
        return RandomForestClassifier(
            n_estimators=RF_PARAMS["n_estimators"],
            max_depth=RF_PARAMS["max_depth"],
            min_samples_split=RF_PARAMS["min_samples_split"],
            min_samples_leaf=RF_PARAMS["min_samples_leaf"],
            random_state=RF_PARAMS["random_state"],
            n_jobs=-1,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv_folds: int = CV_FOLDS,
    ):
        """
        Train Random Forest with stratified cross-validation.
        """

        self.model = self.build_model()

        print(f"[random_forest] Running {cv_folds}-fold CV...")

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
            "[random_forest] CV Accuracy: "
            f"{self.cv_scores.mean():.4f} ± {self.cv_scores.std():.4f}"
        )

        self.model.fit(X_train, y_train)

        print("[random_forest] Training complete.")

        return self.model