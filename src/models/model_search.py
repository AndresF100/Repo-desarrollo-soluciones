
import mlflow
from .pipeline import ModelPipeline
from .model_config import get_param_combinations
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GridSearch:
    def __init__(self, model_name, X_train, y_train, X_val, y_val):
        self.model_name = model_name
        self.param_combinations = get_param_combinations(model_name)
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

    def run(self):
        best_score = -float("inf")
        best_params = None

        for params in self.param_combinations:
            logging.info(f"\t🔎 Probando {self.model_name} con {params}")

            with mlflow.start_run():
                pipeline = ModelPipeline(self.model_name, params)
                model, metrics = pipeline.run(self.X_train, self.y_train, self.X_val, self.y_val)

                mlflow.log_params(params)
                mlflow.log_metric("val_score", metrics["accuracy"])

                if metrics["accuracy"] > best_score:
                    best_score = metrics["accuracy"]
                    best_params = params
                    mlflow.sklearn.log_model(model, f"best_{self.model_name}")

        logging.info(f"\t✅ Mejor modelo: {self.model_name} con {best_params}, score: {best_score}")
