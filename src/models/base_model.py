"""
base_model.py
-------------
Defines the common base class for all machine learning models.

All model classes should inherit from BaseModel to ensure a consistent
interface for training, prediction, and model persistence.
"""

from abc import ABC, abstractmethod
import joblib


class BaseModel(ABC):
    """
    Abstract base class for machine learning models.

    Child classes must implement the build_model() method.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def build_model(self):
        """
        Create and return the machine learning model instance.

        This method must be implemented by each child model class.
        """
        pass

    def train(self, X_train, y_train):
        """
        Train the model using training data.
        """
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        """
        Generate class predictions.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Generate class probability predictions if supported.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        raise AttributeError(
            f"{self.model_name} does not support predict_proba()."
        )

    def save(self, filepath: str):
        """
        Save trained model to disk.
        """
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        """
        Load trained model from disk.
        """
        self.model = joblib.load(filepath)
        return self.model