
from sklearn.metrics import accuracy_score, f1_score

class ModelEvaluator:
    @staticmethod
    def evaluate(model, X, y):
        predictions = model.predict(X)
        return {
            "accuracy": accuracy_score(y, predictions),
            "f1_score": f1_score(y, predictions, average="weighted")
        }
