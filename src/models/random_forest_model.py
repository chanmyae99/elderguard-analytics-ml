"""
Random Forest model implementation.
"""

from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel
from src.utils.config import RF_PARAMS


class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__("Random Forest")

    def build_model(self):
        return RandomForestClassifier(**RF_PARAMS)
