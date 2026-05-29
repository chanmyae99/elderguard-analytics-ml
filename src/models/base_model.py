"""
base_model.py
-------------
Abstract base class that all three ML models inherit from.

Enforces a consistent interface (train, predict, predict_proba)
across Logistic Regression, Random Forest, and XGBoost so they
can be used interchangeably in training_service.py.

Author: [Your Name]
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all ElderGuard ML models.

    All models must implement train(), predict(), and predict_proba().
    This ensures consistent behaviour across models in the pipeline.
    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray,
              cv_folds: int, random_state: int):
        """Fit the model on training data with cross-validation."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""
        pass
