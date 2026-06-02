"""
training_service.py
-------------------
Coordinates training and saving of all machine learning models.
"""

import os

from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.utils.config import MODEL_DIR


class TrainingService:
    """
    Trains all baseline models and saves them to disk.
    """

    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)

        self.models = {
            "logistic_regression": LogisticRegressionModel(),
            "random_forest": RandomForestModel(),
            "xgboost": XGBoostModel(),
        }

    def train_all(self, data):
        """
        Train all models using prepared data from DataService.

        Args:
            data (dict): Output from DataService.prepare()

        Returns:
            dict: Trained model objects.
        """

        trained_models = {}

        print("\n[training_service] Starting model training...")

        # Logistic Regression uses scaled features
        lr_model = self.models["logistic_regression"]
        lr_model.train(
            data["X_train_lr"],
            data["y_train"]
        )
        lr_model.save(
            os.path.join(MODEL_DIR, "logistic_regression.pkl")
        )
        trained_models["logistic_regression"] = lr_model

        # Random Forest uses unscaled tree features
        rf_model = self.models["random_forest"]
        rf_model.train(
            data["X_train_tree"],
            data["y_train"]
        )
        rf_model.save(
            os.path.join(MODEL_DIR, "random_forest.pkl")
        )
        trained_models["random_forest"] = rf_model

        # XGBoost uses unscaled tree features
        xgb_model = self.models["xgboost"]
        xgb_model.train(
            data["X_train_tree"],
            data["y_train"]
        )
        xgb_model.save(
            os.path.join(MODEL_DIR, "xgboost.pkl")
        )
        trained_models["xgboost"] = xgb_model

        print("[training_service] All models trained successfully.")

        return trained_models
    
from src.services.data_service import DataService
from src.services.training_service import TrainingService

data = DataService(apply_imbalance_handling=True).prepare()

trainer = TrainingService()
models = trainer.train_all(data)