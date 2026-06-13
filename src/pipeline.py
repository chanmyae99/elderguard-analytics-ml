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
    
    This orchestrator centralizes execution order and state propagation 
    across data manipulation, feature engineering, modeling, and evaluation phases.
    """

    def __init__(self, apply_imbalance_handling: bool = True):
        """
        Initializes the orchestration pipeline with configurable sampling policies.

        Args:
            apply_imbalance_handling (bool): If True, applies synthetic or resampling techniques 
                                             to skewed target distributions within the DataService.
        """
        self.apply_imbalance_handling = apply_imbalance_handling

    def run(self):
        """
        Execute the complete ML pipeline sequentially.
        
        Data is read from disk, transformed, passed through memory across service classes,
        and finally serialized along with metric evaluation visualizations.

        Returns:
            pd.DataFrame: A comparative matrix summarizing metrics for all tested models.
        """

        print("\n========== ElderGuard ML Pipeline Started ==========\n")

        # Preprocess raw sensor data and generate the cleaned dataset
        print("[pipeline] Step 1: Preprocessing raw data...")
        DataPreprocessor().run()

        # Step 2: Load tables from disk, split data, apply scaling/imbalance handling, and return in-memory dict
        print("\n[pipeline] Step 2: Preparing model-ready data...")
        data = DataService(
            apply_imbalance_handling=self.apply_imbalance_handling
        ).prepare()

        # Step 3: Fit multiple algorithms using the engineered matrices
        print("\n[pipeline] Step 3: Training models...")
        models = TrainingService().train_all(data)

        # Step 4: # Evaluate model performance and generate reports and visualizations
        print("\n[pipeline] Step 4: Evaluating models...")
        results = EvaluationService().evaluate_all(
            models,
            data
        )

        print("\n========== ElderGuard ML Pipeline Completed ==========\n")

        print(results)

        return results