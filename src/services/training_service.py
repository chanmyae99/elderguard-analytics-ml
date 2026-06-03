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

from src.utils.config import (
    RANDOM_STATE,
    REPORT_DIR,
    TUNING_CONFIG,
)


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
    
<<<<<<< HEAD
=======

    def tune_random_forest(self, data):
        """
        Tune Random Forest hyperparameters using RandomizedSearchCV.

        RandomizedSearchCV is used instead of GridSearchCV because it is
        faster and more suitable for large search spaces.

        Weighted F1-score is used because the Activity Level classes are
        imbalanced.
        """

        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import RandomForestClassifier

        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            param_distributions=TUNING_CONFIG["random_forest"],
            n_iter=TUNING_CONFIG["n_iter"],
            scoring=TUNING_CONFIG["scoring"],
            cv=TUNING_CONFIG["cv_folds"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        search.fit(
            data["X_train_tree"],
            data["y_train"]
        )

        self._save_tuning_results(
            search,
            "random_forest_tuning_results.csv"
        )

        print("[training_service] Best RF params:", search.best_params_)
        print("[training_service] Best RF score:", search.best_score_)

        return search.best_params_
    
    
    def tune_xgboost(self, data):
        """
        Tune XGBoost hyperparameters using RandomizedSearchCV.

        XGBoost is tuned separately because it has different parameters
        from Random Forest, such as learning_rate and colsample_bytree.
        """

        from sklearn.model_selection import RandomizedSearchCV
        from xgboost import XGBClassifier

        search = RandomizedSearchCV(
            estimator=XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
            ),
            param_distributions=TUNING_CONFIG["xgboost"],
            n_iter=TUNING_CONFIG["n_iter"],
            scoring=TUNING_CONFIG["scoring"],
            cv=TUNING_CONFIG["cv_folds"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        search.fit(
            data["X_train_tree"],
            data["y_train"]
        )

        self._save_tuning_results(
            search,
            "xgboost_tuning_results.csv"
        )

        print("[training_service] Best XGB params:", search.best_params_)
        print("[training_service] Best XGB score:", search.best_score_)

        return search.best_params_
    
    
    
>>>>>>> feature/training-service
