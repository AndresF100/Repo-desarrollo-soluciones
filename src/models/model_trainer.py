
from .model_registry import ModelRegistry

import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train_and_log(self, X_train, y_train, X_val, y_val):
        self.model.train(X_train, y_train)
        metrics = self.model.evaluate(X_val, y_val)

        ModelRegistry.log_model(self.model, self.model.__class__.__name__, self.model.params, metrics)

        return metrics
