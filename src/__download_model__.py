# Descripción: Script para descargar un modelo de MLflow y empaquetarlo con transformaciones previas.

import mlflow
import os
import shutil
import sys
from dotenv import load_dotenv
from mlflow.pyfunc import PythonModel, save_model

# Asegurar rutas para importar módulos
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_path)

upper_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(upper_path)


# Cargar variables desde el archivo .env
dotenv_path = os.path.join(base_path,"..", ".env.mlflow_server")
load_dotenv(dotenv_path)

# Importar la función de transformación
from modelo_triage.utils.transform_input import prepare_input_data

# Establecer la URI de MLflow
mlflow.set_tracking_uri(f"http://{os.getenv('MLFLOW_MACHINE_IP')}:8050")

# 📂 Ajustar el directorio de salida en la ruta actual
OUTPUT_DIR = os.path.join(base_path, "model_package")

# Cargar el modelo desde MLflow
model_uri = f"models:/{os.getenv('MLFLOW_MODEL_NAME')}/latest"
loaded_model = mlflow.pyfunc.load_model(model_uri)

print(f"✅ Modelo cargado desde MLflow: {model_uri}")


class CustomModelWrapper(PythonModel):
    """
    Wrapper para empaquetar el modelo con transformaciones previas.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, context, input_df):
        # Aplicar transformaciones de entrada
        transformed_df = prepare_input_data(input_df)
        # Realizar predicciones
        return self.model.predict(transformed_df)


# 📦 Eliminar el modelo si ya existe para permitir reemplazo
if os.path.exists(OUTPUT_DIR):
    print(f"♻️ Eliminando modelo existente en: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

# Crear el directorio y empaquetar el modelo con las transformaciones
os.makedirs(OUTPUT_DIR)
save_model(
    path=OUTPUT_DIR,
    python_model=CustomModelWrapper(loaded_model)
)

print(f"✅ Modelo empaquetado en: {OUTPUT_DIR}")
