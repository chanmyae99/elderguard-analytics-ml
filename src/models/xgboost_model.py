"""
XGBoost model implementation.
"""

from xgboost import XGBClassifier

from src.models.base_model import BaseModel
from src.utils.config import XGB_PARAMS


class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")

    def build_model(self):
        return XGBClassifier(**XGB_PARAMS)
