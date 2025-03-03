import mlflow
import warnings
import json

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


class ModelRegistry:
    @staticmethod
    def log_model(model, model_name, params, metrics):
        mlflow.log_params(params)

        # Registrar solo métricas que sean flotantes
        float_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(float_metrics)

        # Guardar artefactos para métricas complejas (listas o diccionarios)
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                with open(f"{key}.json", "w") as f:
                    json.dump(value, f)
                mlflow.log_artifact(f"{key}.json")

        # Guardar el modelo
        mlflow.sklearn.log_model(model, model_name)

    @staticmethod
    def load_model(model_uri):
        return mlflow.sklearn.load_model(model_uri)