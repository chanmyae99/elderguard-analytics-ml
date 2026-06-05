"""
tuning_pipeline.py
------------------
Runs hyperparameter tuning separately from the main ML pipeline.

This pipeline is for experimentation only.

It does not replace the main pipeline. After tuning, the best
hyperparameters should be copied into config/config.yaml under
the models section. Then the main pipeline should be run again
to evaluate all three tuned models fairly.
"""

from src.preprocessing.data_preprocessor import DataPreprocessor
from src.services.data_service import DataService
from src.services.training_service import TrainingService


class TuningPipeline:
    """
    Coordinates model hyperparameter tuning.
    """

    def run(self):
        """
        Execute the tuning workflow.
        """

        print("\n========== ElderGuard Tuning Pipeline Started ==========\n")

        print("[tuning_pipeline] Step 1: Preprocessing raw data...")
        DataPreprocessor().run()

        print("\n[tuning_pipeline] Step 2: Preparing model-ready data...")
        data = DataService(
            apply_imbalance_handling=False
        ).prepare()

        trainer = TrainingService()

        print("\n[tuning_pipeline] Step 3: Tuning Random Forest...")
        best_rf_params = trainer.tune_random_forest(data)

        print("\n[tuning_pipeline] Step 4: Tuning XGBoost...")
        best_xgb_params = trainer.tune_xgboost(data)

        print("\n========== Tuning Completed ==========\n")

        print("Best Random Forest Parameters:")
        print(best_rf_params)

        print("\nBest XGBoost Parameters:")
        print(best_xgb_params)

        print(
            "\nPlease update config/config.yaml with the best parameters, "
            "then run: python -m src.main"
        )


def main():
    pipeline = TuningPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()