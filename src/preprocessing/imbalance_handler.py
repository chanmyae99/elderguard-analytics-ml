"""
imbalance_handler.py
--------------------
Handles class imbalance in the training set.

Dataset imbalance:
  Low Activity      ~57%
  Moderate Activity ~32%
  High Activity     ~11%

Strategy: SMOTE is NOT applied blindly.
  1. Train baseline models first
  2. Evaluate minority-class (High Activity) performance
  3. Enable SMOTE in config only if High Activity recall is weak

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from src.utils.config import RANDOM_STATE


class ImbalanceHandler:
    """
    Optionally applies SMOTE oversampling to the training set.

    Should only ever be applied to training data — never test data.

    Attributes
    ----------
    enabled : bool
        Whether SMOTE is applied. Default False (baseline first).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, enabled: bool = False,
                 random_state: int = RANDOM_STATE):
        self.enabled      = enabled
        self.random_state = random_state

    def handle(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Apply SMOTE to the training set if enabled.

        Args:
            X_train: Training feature matrix.
            y_train: Training target labels.

        Returns:
            X_resampled, y_resampled
        """
        if not self.enabled:
            print("[imbalance_handler] SMOTE disabled — "
                  "using original class distribution.")
            return X_train, y_train

        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        print(f"[imbalance_handler] SMOTE applied.")
        print(f"  Before: {len(X_train):,} → After: {len(X_res):,} samples")

        return pd.DataFrame(X_res, columns=X_train.columns), y_res
