import mlflow
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


class ModelRegistry:
    @staticmethod
    def log_model(model, model_name, params, metrics):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)

    @staticmethod
    def load_model(model_uri):
        return mlflow.sklearn.load_model(model_uri)