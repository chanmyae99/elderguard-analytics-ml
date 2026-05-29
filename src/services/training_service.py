"""
training_service.py
-------------------
Orchestrates training of all three ML models.

Receives prepared data from DataService, instantiates each model,
trains with cross-validation, and returns fitted models for
evaluation by EvaluationService.

"""

import joblib
import os

from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.utils.config import (
    CV_FOLDS,
    RANDOM_STATE,
    MODEL_DIR,
)


class TrainingService:
    """
    Trains all three models and saves them to saved_model/.

    Attributes
    ----------
    models : dict
        Fitted model instances keyed by model name.
    """

    def __init__(self):
        self.models = {}

    def train_all(self, data: dict) -> dict:
        """
        Train Logistic Regression, Random Forest, and XGBoost.

        Args:
            data: Dict returned by DataService.prepare() containing
                  X_train_tree, X_train_lr, y_train etc.

        Returns:
            dict of fitted model instances keyed by name.
        """
        X_train_tree = data["X_train_tree"]
        X_train_lr   = data["X_train_lr"]
        y_train      = data["y_train"]

        # ── Logistic Regression ───────────────────────────────
        print("\n[training_service] Training Logistic Regression ...")
        lr = LogisticRegressionModel()
        lr.train(X_train_lr, y_train,
                 cv_folds=CV_FOLDS, random_state=RANDOM_STATE)
        self.models["Logistic Regression"] = lr
        self._save(lr.model, "logistic_regression_model.pkl")

        # ── Random Forest ─────────────────────────────────────
        print("\n[training_service] Training Random Forest ...")
        rf = RandomForestModel()
        rf.train(X_train_tree, y_train,
                 cv_folds=CV_FOLDS, random_state=RANDOM_STATE)
        self.models["Random Forest"] = rf
        self._save(rf.model, "random_forest_model.pkl")

        # ── XGBoost ───────────────────────────────────────────
        print("\n[training_service] Training XGBoost ...")
        xgb = XGBoostModel()
        xgb.train(X_train_tree, y_train,
                  cv_folds=CV_FOLDS, random_state=RANDOM_STATE)
        self.models["XGBoost"] = xgb
        self._save(xgb.model, "xgboost_model.pkl")

        return self.models

    def _save(self, model, filename: str):
        """Save a fitted model to saved_model/."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump(model, path)
        print(f"[training_service] Model saved → {path}")

    def load(self, filename: str):
        """
        Load a saved model from saved_model/.

        Args:
            filename: e.g. 'xgboost_model.pkl'

        Returns:
            Fitted model.
        """
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model at '{path}'.")
        model = joblib.load(path)
        print(f"[training_service] Model loaded ← {path}")
        return model
