"""
preprocessor.py
---------------
Handles all data cleaning and feature engineering steps for the
ElderGuard gas monitoring dataset:

  1. Remove duplicate rows
  2. Standardise inconsistent categorical labels
  3. Detect and replace temperature sensor outliers
  4. Impute missing values (median for numeric, mode for categorical)
  5. Drop non-predictive identifier columns
  6. Encode categorical features with LabelEncoder

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────────────────────
# Label standardisation helpers
# ──────────────────────────────────────────────────────────────

def _standardise_activity_label(label) -> str:
    """
    Collapse variant spellings of activity-level labels into three
    canonical classes: 'Low Activity', 'Moderate Activity', 'High Activity'.

    Variants observed in the raw data:
        LowActivity, Low_Activity  → Low Activity
        ModerateActivity           → Moderate Activity

    Parameters
    ----------
    label : str or float
        Raw label value (NaN-safe).

    Returns
    -------
    str
        Standardised label, or np.nan if input is NaN.
    """
    if pd.isna(label):
        return np.nan
    label_clean = str(label).strip().lower().replace("_", " ")
    if "low" in label_clean:
        return "Low Activity"
    if "moderate" in label_clean:
        return "Moderate Activity"
    if "high" in label_clean:
        return "High Activity"
    return label  # fallback — leave unchanged


def _standardise_lowercase_col(series: pd.Series) -> pd.Series:
    """Lowercase-strip a string column and replace spaces with underscores."""
    return series.str.strip().str.lower().str.replace(" ", "_", regex=False)


# ──────────────────────────────────────────────────────────────
# Main preprocessing class
# ──────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Stateful preprocessor that fits label encoders on training data
    and applies the same transformation to unseen data.

    Attributes
    ----------
    label_encoders : dict
        Fitted LabelEncoder objects keyed by column name.
    target_encoder : LabelEncoder
        Fitted encoder for the 'Activity Level' target column.
    feature_medians : dict
        Median values computed from training data for numeric imputation.
    feature_modes : dict
        Mode values computed from training data for categorical imputation.
    temp_median : float
        Median temperature (valid range only) used to replace outliers.
    """

    TARGET_COL = "Activity Level"
    CATEGORICAL_FEATURES = ["Time of Day", "HVAC Operation Mode", "Ambient Light Level"]

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Preprocessing sub-section of the YAML config, containing keys:
            temperature_min, temperature_max, median_impute_cols,
            mode_impute_cols, drop_cols.
        """
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoder = LabelEncoder()
        self.feature_medians: Dict[str, float] = {}
        self.feature_modes: Dict[str, str] = {}
        self.temp_median: float = None

    # ── public API ────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all deterministic cleaning steps that do not require
        fitting (duplicate removal, label standardisation, column drops).

        Parameters
        ----------
        df : pd.DataFrame
            Raw data as returned by data_loader.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame (copy of input).
        """
        df = df.copy()

        # Step 1 — Remove duplicate rows
        n_before = len(df)
        df.drop_duplicates(inplace=True)
        print(f"[preprocessor] Removed {n_before - len(df)} duplicate rows "
              f"({len(df):,} rows remain)")

        # Step 2 — Standardise Activity Level labels
        df[self.TARGET_COL] = df[self.TARGET_COL].apply(_standardise_activity_label)
        print(f"[preprocessor] Activity Level classes after standardisation:\n"
              f"  {df[self.TARGET_COL].value_counts().to_dict()}")

        # Step 3 — Standardise other categorical columns (case / spacing)
        for col in ["HVAC Operation Mode", "Ambient Light Level", "Time of Day"]:
            df[col] = _standardise_lowercase_col(df[col])

        # Step 4 — Drop non-predictive columns
        drop_cols = [c for c in self.config["drop_cols"] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)
        print(f"[preprocessor] Dropped columns: {drop_cols}")

        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit imputers and encoders on the training partition, then transform.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned training data (output of self.clean).

        Returns
        -------
        X : pd.DataFrame
            Encoded feature matrix.
        y : pd.Series
            Encoded target labels.
        """
        df = df.copy()

        # ── Temperature outlier replacement (fit phase: compute median) ──
        t_min = self.config["temperature_min"]
        t_max = self.config["temperature_max"]
        valid_mask = df["Temperature"].between(t_min, t_max)
        n_outliers = (~valid_mask).sum()
        self.temp_median = df.loc[valid_mask, "Temperature"].median()
        df.loc[~valid_mask, "Temperature"] = self.temp_median
        print(f"[preprocessor] Temperature outliers replaced: {n_outliers} "
              f"(valid range [{t_min}, {t_max}]°C → median={self.temp_median:.2f})")

        # ── Numeric imputation (fit: compute medians) ──
        for col in self.config["median_impute_cols"]:
            if col in df.columns:
                self.feature_medians[col] = df[col].median()
                df[col] = df[col].fillna(self.feature_medians[col])

        # ── Categorical imputation (fit: compute modes) ──
        for col in self.config["mode_impute_cols"]:
            if col in df.columns:
                self.feature_modes[col] = df[col].mode()[0]
                df[col] = df[col].fillna(self.feature_modes[col])

        print(f"[preprocessor] Remaining nulls after imputation: "
              f"{df.isnull().sum().sum()}")

        # ── Encode categorical features ──
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # ── Encode target ──
        y_raw = df.pop(self.TARGET_COL)
        y = pd.Series(
            self.target_encoder.fit_transform(y_raw.astype(str)),
            name=self.TARGET_COL
        )

        print(f"[preprocessor] Class mapping: "
              f"{ {i: c for i, c in enumerate(self.target_encoder.classes_)} }")

        return df, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply fitted transformations to a held-out dataset (e.g., test set).

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned data (output of self.clean) — must NOT overlap with
            the data used in fit_transform.

        Returns
        -------
        X : pd.DataFrame
            Encoded feature matrix.
        y : pd.Series
            Encoded target labels.
        """
        df = df.copy()

        # Temperature outlier replacement using fitted median
        t_min = self.config["temperature_min"]
        t_max = self.config["temperature_max"]
        mask = ~df["Temperature"].between(t_min, t_max)
        df.loc[mask, "Temperature"] = self.temp_median

        # Numeric imputation using fitted medians
        for col, med in self.feature_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        # Categorical imputation using fitted modes
        for col, mode in self.feature_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode)

        # Encode categorical features using fitted encoders
        for col, le in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda v: v if v in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        # Encode target
        y_raw = df.pop(self.TARGET_COL)
        y = pd.Series(
            self.target_encoder.transform(y_raw.astype(str)),
            name=self.TARGET_COL
        )

        return df, y
