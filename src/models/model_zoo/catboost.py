from catboost import CatBoostClassifier
from ..base_model import BaseModel
from ..model_evaluator import ModelEvaluator

class CatBoostModel(BaseModel):
    def __init__(self, params=None):
        super().__init__(params)

        # Remover cualquier parámetro conflictivo
        for param in ["class_weights", "scale_pos_weight", "auto_class_weights"]:
            self.params.pop(param, None)

        # Establecer auto_class_weights si no está definido
        self.params.setdefault("auto_class_weights", "Balanced")

        # Inicializar el modelo
        self.model = CatBoostClassifier(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return ModelEvaluator.evaluate(self, X, y)
