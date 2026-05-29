"""
evaluation_service.py
---------------------
Evaluates all trained models and saves reports and plots.

Metrics reported
----------------
Accuracy   — overall correctness
Precision  — correctness of positive predictions per class
Recall     — ability to detect all instances of each class
F1-score   — harmonic mean of precision and recall
Confusion Matrix — full prediction breakdown per class

Why F1 and Recall matter here
------------------------------
The dataset is imbalanced (High Activity ~11%). Accuracy alone
is misleading — a model ignoring High Activity still scores ~89%.
F1 Macro and Recall expose this by treating all classes equally.
In an elderly care context, missing a High Activity event has
serious consequences.


"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

from src.utils.config import MODEL_DIR, REPORT_DIR


class EvaluationService:
    """
    Evaluates fitted models on the test set and saves artefacts.

    Attributes
    ----------
    output_dir : str
        Directory for saved plots and reports.
    """

    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(REPORT_DIR, exist_ok=True)

    def evaluate_all(self, models: dict, data: dict,
                     class_names: list) -> list:
        """
        Evaluate all models and print a comparison summary.

        Args:
            models: Dict of fitted model instances from TrainingService.
            data:   Dict from DataService.prepare().
            class_names: Human-readable class labels.

        Returns:
            List of metric dicts — one per model.
        """
        results = []

        for name, model in models.items():
            # Logistic Regression uses scaled test data
            X_test = (data["X_test_lr"]
                      if name == "Logistic Regression"
                      else data["X_test_tree"])

            result = self.evaluate(
                model, X_test, data["y_test"],
                class_names=class_names,
                model_name=name,
                cv_scores=model.cv_scores,
            )
            results.append(result)

        self._print_comparison(results)
        return results

    def evaluate(self, model, X_test: pd.DataFrame,
                 y_test: np.ndarray, class_names: list,
                 model_name: str, cv_scores: np.ndarray) -> dict:
        """
        Compute metrics, print report, and save plots for one model.

        Args:
            model:       Fitted model with .predict() method.
            X_test:      Test feature matrix.
            y_test:      True encoded labels.
            class_names: Human-readable class labels.
            model_name:  Label for filenames and titles.
            cv_scores:   CV accuracy scores from training.

        Returns:
            Dict of evaluation metrics.
        """
        y_pred = model.predict(X_test)

        acc         = accuracy_score(y_test, y_pred)
        precision   = precision_score(y_test, y_pred, average="macro",
                                      zero_division=0)
        recall      = recall_score(y_test, y_pred, average="macro",
                                   zero_division=0)
        f1_macro    = f1_score(y_test, y_pred, average="macro",
                               zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted",
                               zero_division=0)
        report_str  = classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0
        )

        # ── Console output ─────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"  Results — {model_name}")
        print(f"{'='*55}")
        print(f"  Accuracy   : {acc:.4f}")
        print(f"  Precision  : {precision:.4f}  (macro)")
        print(f"  Recall     : {recall:.4f}  (macro)")
        print(f"  F1 Macro   : {f1_macro:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} "
              f"± {cv_scores.std():.4f}")
        print(f"\nClassification Report:\n{report_str}")

        # ── Save text report ───────────────────────────────────
        safe_name   = model_name.lower().replace(" ", "_")
        report_path = os.path.join(REPORT_DIR,
                                   f"{safe_name}_report.txt")
        with open(report_path, "w") as f:
            f.write(f"ElderGuard Analytics — {model_name}\n")
            f.write("=" * 55 + "\n")
            f.write(f"Accuracy   : {acc:.4f}\n")
            f.write(f"Precision  : {precision:.4f} (macro)\n")
            f.write(f"Recall     : {recall:.4f} (macro)\n")
            f.write(f"F1 Macro   : {f1_macro:.4f}\n")
            f.write(f"F1 Weighted: {f1_weighted:.4f}\n")
            f.write(f"CV Accuracy: {cv_scores.mean():.4f} "
                    f"± {cv_scores.std():.4f}\n\n")
            f.write(f"Classification Report:\n{report_str}")
        print(f"[evaluation_service] Report saved → {report_path}")

        # ── Confusion matrix ───────────────────────────────────
        cm   = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {model_name}",
                     fontweight="bold")
        plt.tight_layout()
        cm_path = os.path.join(REPORT_DIR,
                               f"{safe_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"[evaluation_service] Confusion matrix → {cm_path}")

        # ── Feature importance (tree models only) ──────────────
        if hasattr(model, "feature_importances_"):
            self._plot_feature_importance(
                model.feature_importances_,
                X_test.columns.tolist(),
                model_name, safe_name
            )

        return {
            "model_name" : model_name,
            "accuracy"   : acc,
            "precision"  : precision,
            "recall"     : recall,
            "f1_macro"   : f1_macro,
            "f1_weighted": f1_weighted,
            "cv_mean"    : cv_scores.mean(),
            "cv_std"     : cv_scores.std(),
        }

    def _plot_feature_importance(self, importances, feature_names,
                                 model_name, safe_name):
        """Plot and save a horizontal feature importance bar chart."""
        fi_df = (pd.DataFrame({"Feature": feature_names,
                               "Importance": importances})
                   .sort_values("Importance", ascending=True))
        fig, ax = plt.subplots(
            figsize=(10, max(6, len(fi_df) * 0.3))
        )
        ax.barh(fi_df["Feature"], fi_df["Importance"],
                color="#1976D2", edgecolor="white")
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Feature Importance — {model_name}",
                     fontweight="bold")
        plt.tight_layout()
        path = os.path.join(
            REPORT_DIR, f"{safe_name}_feature_importance.png"
        )
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[evaluation_service] Feature importance → {path}")

    def _print_comparison(self, results: list):
        """Print a summary comparison table of all models."""
        print(f"\n{'='*70}")
        print("  MODEL COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} "
              f"{'Recall':>7} {'F1 Mac':>7} {'CV Acc':>7}")
        print("-" * 70)
        for r in results:
            print(f"{r['model_name']:<25} {r['accuracy']:>9.4f} "
                  f"{r['precision']:>10.4f} {r['recall']:>7.4f} "
                  f"{r['f1_macro']:>7.4f} {r['cv_mean']:>7.4f}")
        print(f"{'='*70}")
