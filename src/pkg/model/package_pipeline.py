import mlflow
import os
from dotenv import load_dotenv

# Cargar variables desde el archivo .env ubicado dos niveles arriba
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env.mlflow_server"))
load_dotenv(dotenv_path)

# Establecer la URI de MLflow
mlflow.set_tracking_uri(f"http://{os.getenv('MLFLOW_MACHINE_IP')}:8050")

OUTPUT_DIR = "model_package"

# Cargar modelo desde MLflow
model_uri = f"models:/{os.getenv('MLFLOW_MODEL_NAME')}/latest"
model = mlflow.pyfunc.load_model(model_uri)


# Empaquetar modelo en un directorio
mlflow.pyfunc.save_model(
    dst_path=OUTPUT_DIR,
    python_model=model
)

print(f"Modelo empaquetado en {OUTPUT_DIR}")
