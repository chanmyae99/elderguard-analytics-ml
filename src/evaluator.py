"""
evaluator.py
------------
Evaluates a trained model on the held-out test set, prints a full
classification report, and saves evaluation artefacts:

  * Text report  → saved_model/evaluation_report.txt
  * Confusion matrix plot
  * Feature importance plot (tree-based models only)

Author: [Your Name]
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series,
             class_names: list, train_result: dict,
             output_dir: str, report_filename: str) -> dict:
    """
    Evaluate the fitted model and persist artefacts.

    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
    X_test : pd.DataFrame
        Held-out feature matrix.
    y_test : pd.Series
        True integer-encoded labels.
    class_names : list of str
        Human-readable class labels in label-encoder order.
    train_result : dict
        Output of trainer.train() — provides CV scores.
    output_dir : str
        Directory to write reports and plots.
    report_filename : str
        Filename for the plain-text evaluation report.

    Returns
    -------
    dict
        Evaluation metrics: accuracy, f1_macro, f1_weighted.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)

    acc        = accuracy_score(y_test, y_pred)
    f1_macro   = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report_str = classification_report(y_test, y_pred, target_names=class_names)

    # ── Console output ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Test Accuracy   : {acc:.4f}")
    print(f"  F1 Macro        : {f1_macro:.4f}")
    print(f"  F1 Weighted     : {f1_weighted:.4f}")
    print(f"  CV Accuracy     : {train_result['cv_mean']:.4f} "
          f"± {train_result['cv_std']:.4f}")
    print("\nClassification Report:")
    print(report_str)

    # ── Write text report ──────────────────────────────────────
    report_path = os.path.join(output_dir, report_filename)
    with open(report_path, "w") as f:
        f.write("ElderGuard Analytics — Model Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy   : {acc:.4f}\n")
        f.write(f"F1 Macro        : {f1_macro:.4f}\n")
        f.write(f"F1 Weighted     : {f1_weighted:.4f}\n")
        f.write(f"CV Accuracy     : {train_result['cv_mean']:.4f} "
                f"± {train_result['cv_std']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    print(f"[evaluator] Report saved → {report_path}")

    # ── Confusion matrix ───────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[evaluator] Confusion matrix saved → {cm_path}")

    # ── Feature importance (tree-based models only) ────────────
    _plot_feature_importance(model, X_test.columns.tolist(), output_dir)

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def _plot_feature_importance(model, feature_names: list, output_dir: str):
    """Extract and plot feature importances for tree-based models."""
    # Unwrap Pipeline if necessary
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps.get("clf", model[-1])

    if not hasattr(estimator, "feature_importances_"):
        print("[evaluator] Model does not expose feature_importances_ — skipping plot.")
        return

    importances = estimator.feature_importances_
    fi_df = (pd.DataFrame({"Feature": feature_names, "Importance": importances})
               .sort_values("Importance", ascending=True))

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"],
                   color="#1976D2", edgecolor="white")
    ax.set_xlabel("Feature Importance (Gini / Gain)")
    ax.set_title("Key Features for Activity Level Prediction", fontweight="bold")
    for bar, val in zip(bars, fi_df["Importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"[evaluator] Feature importance plot saved → {fi_path}")
