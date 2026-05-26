"""
model_io.py
-----------
Utilities for persisting and loading trained models to/from disk
using joblib (the recommended serialiser for scikit-learn objects).

Author: [Your Name]
"""

import os
import joblib


def save_model(model, output_dir: str, filename: str) -> str:
    """
    Serialise a fitted model to disk.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
        The trained model to save.
    output_dir : str
        Directory in which to write the file (created if absent).
    filename : str
        File name, e.g. 'model.pkl'.

    Returns
    -------
    str
        Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    joblib.dump(model, path)
    print(f"[model_io] Model saved → {path}")
    return path


def load_model(path: str):
    """
    Deserialise a model from disk.

    Parameters
    ----------
    path : str
        Full path to the .pkl file.

    Returns
    -------
    Fitted sklearn estimator or Pipeline.

    Raises
    ------
    FileNotFoundError
        If no file exists at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at '{path}'.")
    model = joblib.load(path)
    print(f"[model_io] Model loaded ← {path}")
    return model
