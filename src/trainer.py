"""
trainer.py
----------
Handles model training and cross-validation for the ElderGuard
activity-level prediction pipeline.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score


def train(model, X_train: pd.DataFrame, y_train: pd.Series,
          cv_folds: int = 5, scoring: str = "accuracy",
          random_state: int = 42) -> dict:
    """
    Fit a model on the training set and compute stratified k-fold
    cross-validation scores.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
        Unfitted model returned by model_factory.build_model().
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels (integer-encoded).
    cv_folds : int
        Number of stratified folds for cross-validation.
    scoring : str
        Sklearn scoring metric string (e.g. 'accuracy', 'f1_macro').
    random_state : int
        Seed for StratifiedKFold shuffle.

    Returns
    -------
    dict
        Keys: 'model' (fitted), 'cv_scores' (array),
              'cv_mean' (float), 'cv_std' (float).
    """
    print(f"[trainer] Starting cross-validation ({cv_folds}-fold, scoring='{scoring}') ...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring=scoring, n_jobs=-1)
    print(f"[trainer] CV {scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print(f"[trainer] Fitting final model on full training set ...")
    model.fit(X_train, y_train)
    print(f"[trainer] Training complete.")

    return {
        "model": model,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }
