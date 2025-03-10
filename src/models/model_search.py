import mlflow
from .pipeline import ModelPipeline
from .model_config import get_param_combinations
import logging
import warnings
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ocultar advertencias de MLflow
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

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
                # Agregar un tag con el nombre del modelo
                mlflow.set_tag("model_name", self.model_name)

                pipeline = ModelPipeline(self.model_name, params)
                model, metrics = pipeline.run(self.X_train, self.y_train, self.X_val, self.y_val)

                mlflow.log_params(params)
                mlflow.log_metric("val_f1", metrics["f1_score"])
                

                if metrics["f1_score"] > best_score:
                    best_score = metrics["f1_score"]
                    best_params = params

                    # # Convertir csr_matrix a un formato serializable
                    # X_sample = self.X_train[:1].toarray() if hasattr(self.X_train, "toarray") else self.X_train[:1]
                    # input_example = pd.DataFrame(X_sample).to_dict(orient='records')[0]

                    mlflow.sklearn.log_model(model, f"best_{self.model_name}"
                                             #, input_example=input_example
                                             )

        logging.info(f"\t✅ Mejor modelo: {self.model_name} con {best_params}, f1_score: {best_score}")
