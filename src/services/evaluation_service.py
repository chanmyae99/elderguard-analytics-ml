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
matplotlib.use("Agg")  # Non-interactive backend to prevent GUI thread rendering in production/headless servers
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
    Handles model evaluation, comparative performance analysis, and model explainability reporting.
    
    Provides automated artifact generation including structured CSV summaries, 
    formatted classification text reports, and static visualization plots.
    """

    def __init__(self):
        """
        Initializes the evaluation service and sets up the filesystem hierarchy 
        for storing reports and visualization metrics.
        """
        self.metrics_dir = os.path.join(REPORT_DIR, "metrics")
        self.figures_dir = os.path.join(REPORT_DIR, "figures")

        # Ensure target directories exist before any evaluation procedures begin
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def evaluate_all(self, trained_models: dict, data: dict) -> pd.DataFrame:
        """
        Evaluate all trained models using consistent metrics.

        Args:
            trained_models (dict): Dictionary of trained model wrappers containing custom estimator abstractions.
            data (dict): Prepared dataset dictionary containing train/test splits and metadata.

        Returns:
            pd.DataFrame: Summary table containing comparative model performance metrics.
        """

        results = []

        for model_key, model in trained_models.items():

            # Handle architectural discrepancies in feature matrices (e.g., scaled vs. unscaled variants)
            if model_key == "logistic_regression":
                X_test = data["X_test_lr"]
            else:
                X_test = data["X_test_tree"]

            # Compute standard classification metrics and export individual artifacts
            result = self.evaluate_model(
                model_key=model_key,
                model=model,
                X_test=X_test,
                y_test=data["y_test"],
                class_names=data["class_names"],
            )

            results.append(result)

        # Consolidate metrics across models into a single comparative tabular dataframe
        results_df = pd.DataFrame(results)

        summary_path = os.path.join(self.metrics_dir, "model_comparison_summary.csv")

        results_df.to_csv(summary_path, index=False)

        # Automatically flag and serialize the optimal performer based on the designated primary metric
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
        X_test: pd.DataFrame,
        y_test: pd.Series,
        class_names: list,
    ) -> dict:
        """
        Evaluate one model and save its report, confusion matrix,
        and feature importance where available.

        Args:
            model_key (str): Unique identifier for the model being evaluated.
            model (object): Model wrapper object holding the underlying fitted estimator.
            X_test (pd.DataFrame): Evaluation feature matrix.
            y_test (pd.Series): Ground-truth target labels.
            class_names (list): Readable string labels corresponding to internal encoded classes.

        Returns:
            dict: Computed dictionary containing evaluation metrics.
        """

        # Generate model predictions
        y_pred = model.predict(X_test)

        # Compile comprehensive metrics suite; zero_division=0 handles unpredicted labels safely
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

        # Trigger downstream asynchronous/synchronous file export operations
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
        model_key: str,
        y_test: pd.Series,
        y_pred: pd.Series,
        class_names: list,
        metrics: dict,
    ):
        """
        Save classification report and key metric values as text.
        """

        # Generate the standard scikit-learn precision/recall/f1 breakdown per class
        report = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )

        file_path = os.path.join(
            self.metrics_dir, f"{model_key}_classification_report.txt"
        )

        # Export human-readable model performance audit log
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
        model_key: str,
        y_test: pd.Series,
        y_pred: pd.Series,
        class_names: list,
    ):
        """
        Save confusion matrix plot for model evaluation.
        """

        # Build visual confusion matrix using matplotlib backend engine
        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=class_names,
            xticks_rotation=45,
        )

        display.ax_.set_title(f"{model_key.replace('_', ' ').title()} Confusion Matrix")

        file_path = os.path.join(self.figures_dir, f"{model_key}_confusion_matrix.png")

        # Guard against cut-off text labels on the peripheral bounds of the figure axis
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()  # Clean up memory allocation from active matplotlib registers

        print(f"[evaluation_service] Saved confusion matrix: {file_path}")

    def _save_feature_importance(
        self,
        model_key: str,
        model,
        feature_names: pd.Index,
        top_n: int = 15,
    ):
        """
        Save feature importance values and plot.

        Tree-based models use feature_importances_.
        Logistic Regression uses mean absolute coefficients.
        """

        # Access underlying base estimator instance inside the custom wrapper
        estimator = model.model

        # Extract importances based on architectural implementation properties
        if hasattr(estimator, "feature_importances_"):
            importance_values = estimator.feature_importances_

        elif hasattr(estimator, "coef_"):
            # Use mean absolute coefficient weight across classes for multi-class support
            importance_values = abs(estimator.coef_).mean(axis=0)

        else:
            # Fallback gracefully for algorithm architectures that do not support raw intrinsic importance profiles
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

        # Isolate the top N features to plot
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 6))
        
        # [::-1] Inverts the dataframe slices to plot the highest importance metric at the top of the horizontal bar chart
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

        # Locate the index row representing the maximum value for the specified evaluation metric
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

        # Serialize optimal model artifact to disk for inference deployment
        joblib.dump(
            best_model,
            best_model_path
        )

        # Document operational metadata summary for the selected champion model
        with open(best_model_info_path, "w", encoding="utf-8") as file:
            file.write("Best Model Selection\n")
            file.write("=" * 50 + "\n\n")
            file.write(f"Selected Metric: {selection_metric}\n")
            file.write(f"Best Model: {best_model_name}\n\n")

            file.write("Best Model Scores\n")
            file.write("-" * 50 + "\n")

            # Dynamic type check to safely format float values while retaining text identifiers intact
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