"""
Logistic Regression model implementation.
"""

from sklearn.linear_model import LogisticRegression

from src.models.base_model import BaseModel
from src.utils.config import LR_PARAMS


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__("Logistic Regression")

    def build_model(self):
        return LogisticRegression(**LR_PARAMS)

