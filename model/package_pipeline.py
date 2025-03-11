import mlflow
import os
import pandas as pd
import sys
from dotenv import load_dotenv
from mlflow.pyfunc import PythonModel, save_model

# Asegurar rutas para importar módulos
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

# Cargar variables desde el archivo .env
dotenv_path = os.path.join(base_path, ".env.mlflow_server")
load_dotenv(dotenv_path)

# Importar la función de transformación
from model.transform_input import prepare_input_data

# Establecer la URI de MLflow
mlflow.set_tracking_uri(f"http://{os.getenv('MLFLOW_MACHINE_IP')}:8050")

# Directorio de salida del modelo empaquetado
OUTPUT_DIR = "model_package"

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

    def predict(self, context, input_df: pd.DataFrame):
        # Aplicar transformaciones de entrada
        transformed_df = prepare_input_data(input_df)
        # Realizar predicciones
        return self.model.predict(transformed_df)

# si no existe el modelo empaquetado, lo creamos
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    # Empaquetar el modelo con las transformaciones
    save_model(
        path=OUTPUT_DIR,
        python_model=CustomModelWrapper(loaded_model)
    )

    print(f"✅ Modelo empaquetado en: {OUTPUT_DIR}")

else:
    print(f"✅ Modelo empaquetado ya existe en: {OUTPUT_DIR}")
    # Cargar el modelo empaquetado
    loaded_model = mlflow.pyfunc.load_model(OUTPUT_DIR)



def predict_with_mlflow_model(input_df: pd.DataFrame):
    """
    Realiza predicciones utilizando el modelo empaquetado.
    """
    if input_df.empty:
        raise ValueError("El DataFrame de entrada está vacío.")

    # Realizar predicciones
    predictions = loaded_model.predict(input_df)
    return predictions

# test
# df = pd.read_csv("data/user_input_example/user_input.csv", sep=";")
# predictions = predict_with_mlflow_model(df)
# print(predictions)
