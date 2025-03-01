
from .model_trainer import ModelTrainer
from .model_zoo.random_forest import RandomForestModel
from .model_zoo.xgboost_model import XGBoostModel

class ModelPipeline:
    def __init__(self, model_name, params=None):
        self.model_name = model_name
        self.params = params or {}

    def get_model(self):
        if self.model_name == "RandomForest":
            return RandomForestModel(self.params)
        elif self.model_name == "XGBoost":
            return XGBoostModel(self.params)
        else:
            raise ValueError(f"Modelo {self.model_name} no soportado")

    def run(self, X_train, y_train, X_val, y_val):
        model = self.get_model()
        trainer = ModelTrainer(model)
        metrics = trainer.train_and_log(X_train, y_train, X_val, y_val)
        return model, metrics
