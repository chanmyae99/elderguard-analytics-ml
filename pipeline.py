"""
pipeline.py
-----------
End-to-end ML pipeline for ElderGuard Analytics.

Wires together all services in the correct order:
  DataService → TrainingService → EvaluationService

All configuration is read from config/config.yaml via
src/utils/config.py

"""

from src.services.data_service import DataService
from src.services.training_service import TrainingService
from src.services.evaluation_service import EvaluationService


def run_pipeline(apply_imbalance_handling: bool = False):
    """
    Execute the full training and evaluation pipeline.

    Args:
        apply_imbalance_handling: Set True to apply SMOTE on
            training data. Default False — evaluate baseline first.
    """
    print(f"\n{'='*55}")
    print("  ElderGuard Analytics — ML Pipeline")
    print("  Models: Logistic Regression · Random Forest · XGBoost")
    print(f"{'='*55}")

    # ── Step 1: Load and prepare data ─────────────────────────
    print("\n[Step 1] Preparing data ...")
    data_service = DataService(
        apply_imbalance_handling=apply_imbalance_handling
    )
    data = data_service.prepare()

    # ── Step 2: Train all models ───────────────────────────────
    print("\n[Step 2] Training models ...")
    training_service = TrainingService()
    models = training_service.train_all(data)

    # ── Step 3: Evaluate all models ────────────────────────────
    print("\n[Step 3] Evaluating models ...")
    evaluation_service = EvaluationService()
    results = evaluation_service.evaluate_all(
        models=models,
        data=data,
        class_names=data_service.class_names,
    )

    print(f"\n{'='*55}")
    print("  Pipeline complete.")
    print("  Reports → reports/")
    print("  Models  → saved_model/")
    print(f"{'='*55}\n")

    return results
