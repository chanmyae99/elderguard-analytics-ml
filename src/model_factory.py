"""
model_factory.py
----------------
Factory module that instantiates a machine learning model from
the pipeline configuration.

Currently registered models
----------------------------
* xgboost  — gradient boosting; best accuracy on this dataset (~69%)

HOW TO ADD A NEW MODEL
----------------------
1. Import the class at the top of this file.
2. Add an elif block inside build_model() following the same pattern.
3. Add the matching config block to config.yaml under `models:`.

Example — adding Random Forest:

    Step 1 (here):
        from sklearn.ensemble import RandomForestClassifier

    Step 2 (here):
        elif name == "random_forest":
            return RandomForestClassifier(
                n_estimators=model_config.get("n_estimators", 300),
                max_depth=model_config.get("max_depth"),
                min_samples_leaf=model_config.get("min_samples_leaf", 1),
                max_features=model_config.get("max_features", "sqrt"),
                class_weight=model_config.get("class_weight", None),
                n_jobs=model_config.get("n_jobs", -1),
                random_state=random_state,
            )

    Step 3 (config.yaml):
        random_forest:
          n_estimators: 300
          max_depth: null
          min_samples_leaf: 1
          max_features: "sqrt"
          class_weight: "balanced"
          n_jobs: -1

Author: [Your Name]
"""

import xgboost as xgb

# ── Add imports for new models here ──────────────────────────
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler


def build_model(model_name: str, model_config: dict, random_state: int = 42):
    """
    Construct and return an unfitted scikit-learn compatible model
    based on the name and hyperparameters supplied.

    Parameters
    ----------
    model_name : str
        Must match a registered model name (e.g. 'xgboost').
    model_config : dict
        Hyperparameter dict from config.yaml under models.<model_name>.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Unfitted sklearn-compatible estimator.

    Raises
    ------
    ValueError
        If an unrecognised model name is provided.
    """
    name = model_name.lower().strip()

    # ── Model 1: XGBoost ─────────────────────────────────────
    if name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=model_config.get("n_estimators", 400),
            max_depth=model_config.get("max_depth", 8),
            learning_rate=model_config.get("learning_rate", 0.05),
            subsample=model_config.get("subsample", 0.8),
            colsample_bytree=model_config.get("colsample_bytree", 0.8),
            eval_metric="mlogloss",
            n_jobs=model_config.get("n_jobs", -1),
            random_state=random_state,
        )

    # ── Model 2: Add here ─────────────────────────────────────
    # elif name == "random_forest":
    #     return RandomForestClassifier(
    #         n_estimators=model_config.get("n_estimators", 300),
    #         max_depth=model_config.get("max_depth"),
    #         min_samples_leaf=model_config.get("min_samples_leaf", 1),
    #         max_features=model_config.get("max_features", "sqrt"),
    #         class_weight=model_config.get("class_weight", None),
    #         n_jobs=model_config.get("n_jobs", -1),
    #         random_state=random_state,
    #     )

    # ── Model 3: Add here ─────────────────────────────────────
    # elif name == "logistic_regression":
    #     clf = LogisticRegression(
    #         max_iter=model_config.get("max_iter", 1000),
    #         C=model_config.get("C", 1.0),
    #         solver=model_config.get("solver", "lbfgs"),
    #         random_state=random_state,
    #     )
    #     # Logistic Regression requires feature scaling
    #     return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Register it in src/model_factory.py first."
        )
