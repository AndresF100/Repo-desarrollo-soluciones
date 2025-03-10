from .model_trainer import ModelTrainer
from .model_zoo.random_forest import RandomForestModel
from .model_zoo.xgboost_model import XGBoostModel
from .model_zoo.lightgbm import LightGBMModel
from .model_zoo.catboost import CatBoostModel
from .model_zoo.mlp_model import MLPModel

class ModelPipeline:
    def __init__(self, model_name, params=None):
        self.model_name = model_name
        self.params = params or {}

    def get_model(self):
        model_registry = {
            "RandomForest": RandomForestModel,
            "XGBoost": XGBoostModel,
            "LightGBM": LightGBMModel,
            "CatBoost": CatBoostModel,
            "MLP": MLPModel,
        }

        if self.model_name not in model_registry:
            raise ValueError(f"Modelo {self.model_name} no soportado")

        return model_registry[self.model_name](self.params)

    def run(self, X_train, y_train, X_val, y_val):
        model = self.get_model()
        trainer = ModelTrainer(model)
        metrics = trainer.train_and_log(X_train, y_train, X_val, y_val)
        return model, metrics
