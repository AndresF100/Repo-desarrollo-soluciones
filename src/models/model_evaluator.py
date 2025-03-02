import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import json

class ModelEvaluator:
    @staticmethod
    def evaluate(model, X, y):
        predictions = model.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "f1_score": f1_score(y, predictions, average="weighted"),
            "precision": precision_score(y, predictions, average="weighted"),
            "recall": recall_score(y, predictions, average="weighted"),
        }

        # Manejar la matriz de confusión y el reporte como artefactos
        metrics["confusion_matrix"] = confusion_matrix(y, predictions).tolist()
        metrics["classification_report"] = classification_report(y, predictions, output_dict=True)

        return metrics
