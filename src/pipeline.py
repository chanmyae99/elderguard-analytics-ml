"""
pipeline.py
-----------
Orchestrates the complete ElderGuard machine learning pipeline.
"""

from src.preprocessing.data_preprocessor import DataPreprocessor
from src.services.data_service import DataService
from src.services.training_service import TrainingService
from src.services.evaluation_service import EvaluationService


class Pipeline:
    """
    Runs the full ML workflow from raw data preprocessing to evaluation.
    """

    def __init__(self, apply_imbalance_handling: bool = True):
        self.apply_imbalance_handling = apply_imbalance_handling

    def run(self):
        """
        Execute the complete ML pipeline.
        """

        print("\n========== ElderGuard ML Pipeline Started ==========\n")

        print("[pipeline] Step 1: Preprocessing raw data...")
        DataPreprocessor().run()

        print("\n[pipeline] Step 2: Preparing model-ready data...")
        data = DataService(
            apply_imbalance_handling=self.apply_imbalance_handling
        ).prepare()

        print("\n[pipeline] Step 3: Training models...")
        models = TrainingService().train_all(data)

        print("\n[pipeline] Step 4: Evaluating models...")
        results = EvaluationService().evaluate_all(
            models,
            data
        )

        print("\n========== ElderGuard ML Pipeline Completed ==========\n")

        print(results)

        return results