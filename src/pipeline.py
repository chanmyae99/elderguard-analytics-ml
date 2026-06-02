from src.preprocessing.data_preprocessor import DataPreprocessor
from src.services.data_service import DataService
from src.services.training_service import TrainingService
from src.services.evaluation_service import EvaluationService


DataPreprocessor().run()

data = DataService(
    apply_imbalance_handling=True
).prepare()

models = TrainingService().train_all(data)

results = EvaluationService().evaluate_all(
    models,
    data
)

print(results)