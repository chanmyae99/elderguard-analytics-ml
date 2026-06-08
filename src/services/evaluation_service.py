"""
evaluation_service.py
---------------------
Evaluates trained models and saves model performance outputs.

This service supports:
- model metric comparison
- classification reports
- confusion matrix visualizations
- feature importance analysis

This aligns with the project requirement to evaluate at least three
models and identify key features for Activity Level prediction.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from src.utils.config import MODEL_DIR

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.utils.config import REPORT_DIR


class EvaluationService:
    """
    Handles model evaluation and explainability reporting.
    """

    def __init__(self):
        self.metrics_dir = os.path.join(REPORT_DIR, "metrics")
        self.figures_dir = os.path.join(REPORT_DIR, "figures")

        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def evaluate_all(self, trained_models: dict, data: dict):
        """
        Evaluate all trained models using consistent metrics.

        Args:
            trained_models: Dictionary of trained model wrappers.
            data: Prepared dataset returned by DataService.

        Returns:
            pd.DataFrame: Summary table of model metrics.
        """

        results = []

        for model_key, model in trained_models.items():

            if model_key == "logistic_regression":
                X_test = data["X_test_lr"]
            else:
                X_test = data["X_test_tree"]

            result = self.evaluate_model(
                model_key=model_key,
                model=model,
                X_test=X_test,
                y_test=data["y_test"],
                class_names=data["class_names"],
            )

            results.append(result)

        results_df = pd.DataFrame(results)

        summary_path = os.path.join(self.metrics_dir, "model_comparison_summary.csv")

        results_df.to_csv(summary_path, index=False)

        self._save_best_model(
            results_df=results_df,
            trained_models=trained_models,
            selection_metric="f1_weighted",
        )

        print(
            f"[evaluation_service] Saved model comparison summary: " f"{summary_path}"
        )

        return results_df

    def evaluate_model(
        self,
        model_key: str,
        model,
        X_test,
        y_test,
        class_names,
    ):
        """
        Evaluate one model and save its report, confusion matrix,
        and feature importance where available.
        """

        y_pred = model.predict(X_test)

        metrics = {
            "model": model_key,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "recall_weighted": recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),
        }

        print(f"\n[evaluation_service] Evaluating {model_key}")
        print(f"Accuracy    : {metrics['accuracy']:.4f}")
        print(f"Weighted F1 : {metrics['f1_weighted']:.4f}")
        print(f"Macro F1    : {metrics['f1_macro']:.4f}")

        self._save_classification_report(
            model_key,
            y_test,
            y_pred,
            class_names,
            metrics,
        )

        self._save_confusion_matrix(
            model_key,
            y_test,
            y_pred,
            class_names,
        )

        self._save_feature_importance(
            model_key,
            model,
            X_test.columns,
        )

        return metrics

    def _save_classification_report(
        self,
        model_key,
        y_test,
        y_pred,
        class_names,
        metrics,
    ):
        """
        Save classification report and key metric values as text.
        """

        report = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )

        file_path = os.path.join(
            self.metrics_dir, f"{model_key}_classification_report.txt"
        )

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"Model: {model_key}\n")
            file.write("=" * 60 + "\n\n")

            file.write("Overall Metrics\n")
            file.write("-" * 60 + "\n")

            for key, value in metrics.items():
                if key != "model":
                    file.write(f"{key}: {value:.4f}\n")

            file.write("\nClassification Report\n")
            file.write("-" * 60 + "\n")
            file.write(report)

        print(f"[evaluation_service] Saved report: {file_path}")

    def _save_confusion_matrix(
        self,
        model_key,
        y_test,
        y_pred,
        class_names,
    ):
        """
        Save confusion matrix plot for model evaluation.
        """

        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=class_names,
            xticks_rotation=45,
        )

        display.ax_.set_title(f"{model_key.replace('_', ' ').title()} Confusion Matrix")

        file_path = os.path.join(self.figures_dir, f"{model_key}_confusion_matrix.png")

        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

        print(f"[evaluation_service] Saved confusion matrix: {file_path}")

    def _save_feature_importance(
        self,
        model_key,
        model,
        feature_names,
        top_n: int = 15,
    ):
        """
        Save feature importance values and plot.

        Tree-based models use feature_importances_.
        Logistic Regression uses mean absolute coefficients.
        """

        estimator = model.model

        if hasattr(estimator, "feature_importances_"):
            importance_values = estimator.feature_importances_

        elif hasattr(estimator, "coef_"):
            importance_values = abs(estimator.coef_).mean(axis=0)

        else:
            print(
                f"[evaluation_service] Feature importance not available "
                f"for {model_key}."
            )
            return

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance_values,
            }
        ).sort_values(
            by="importance",
            ascending=False,
        )

        csv_path = os.path.join(self.metrics_dir, f"{model_key}_feature_importance.csv")

        importance_df.to_csv(csv_path, index=False)

        print(f"[evaluation_service] Saved feature importance CSV: {csv_path}")

        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(
            top_features["feature"][::-1],
            top_features["importance"][::-1],
        )

        plt.title(f"Top {top_n} Features - " f"{model_key.replace('_', ' ').title()}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")

        plt.tight_layout()

        fig_path = os.path.join(self.figures_dir, f"{model_key}_feature_importance.png")

        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"[evaluation_service] Saved feature importance plot: {fig_path}")


    def _save_best_model(
        self,
        results_df: pd.DataFrame,
        trained_models: dict,
        selection_metric: str = "f1_weighted"
    ):
        """
        Select and save the best model based on the chosen evaluation metric.
        """

        os.makedirs(MODEL_DIR, exist_ok=True)

        best_row = results_df.loc[
        results_df[selection_metric].idxmax()
        ]

        best_model_name = best_row["model"]
        best_model = trained_models[best_model_name]

        best_model_path = os.path.join(
            MODEL_DIR,
            "best_model.pkl"
        )

        best_model_info_path = os.path.join(
            MODEL_DIR,
            "best_model_info.txt"
        )

        joblib.dump(
            best_model,
            best_model_path
        )

        with open(best_model_info_path, "w", encoding="utf-8") as file:
            file.write("Best Model Selection\n")
            file.write("=" * 50 + "\n\n")
            file.write(f"Selected Metric: {selection_metric}\n")
            file.write(f"Best Model: {best_model_name}\n\n")

            file.write("Best Model Scores\n")
            file.write("-" * 50 + "\n")

            for column in results_df.columns:
                value = best_row[column]

                if isinstance(value, float):
                    file.write(f"{column}: {value:.4f}\n")
                else:
                    file.write(f"{column}: {value}\n")

        print(
            f"[evaluation_service] Best model selected: "
            f"{best_model_name}"
        )

        print(
            f"[evaluation_service] Saved best model: "
            f"{best_model_path}"
        )


