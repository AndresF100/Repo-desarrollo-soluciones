from sklearn.neural_network import MLPClassifier
from ..base_model import BaseModel
from ..model_evaluator import ModelEvaluator

class MLPModel(BaseModel):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = MLPClassifier(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return ModelEvaluator.evaluate(self, X, y)
