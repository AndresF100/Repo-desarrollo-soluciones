import numpy as np
import pandas as pd
import mlflow
import scipy.sparse
from src.models.model_search import GridSearch
import logging
from dotenv import load_dotenv
import os

# Carga las variables de entorno
load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Iniciando pipeline de experimentación...")


# 1. Carga de los datos

# Función para cargar datos (maneja tanto sparse como dense)
def load_data(X_path, y_path, is_sparse=True):
    if is_sparse:
        X = scipy.sparse.load_npz(X_path)  # Carga matriz dispersa
    else:
        X = pd.read_parquet(X_path).values  # Alternativa si se usa Parquet
    y = pd.read_csv(y_path).values.ravel()  # Carga el target
    return X, y

X_train, y_train = load_data('data/processed/X_train.npz', 'data/processed/y_train.csv', is_sparse=True)
X_val, y_val = load_data('data/processed/X_val.npz', 'data/processed/y_val.csv', is_sparse=True)


logging.info(f"✅ Datos cargados correctamente")

# 2. Configura la URI de MLflow (local o remoto)
mlflow.set_tracking_uri(f"http://{os.getenv("MLFLOW_MACHINE_IP")}:8050")

experiment_name = os.getenv("EXPERIMENT_NAME")
mlflow.set_experiment(experiment_name)

logging.info(f"✅ MLflow configurado correctamente, experimento: {experiment_name}")


logging.info("🔎 Iniciando experimentación...")
# 3. Ejecuta Grid Search para RandomForest
search = GridSearch("RandomForest", X_train, y_train, X_val, y_val)
search.run()

# 4. Ejecuta Grid Search para XGBoost
search = GridSearch("XGBoost", X_train, y_train, X_val, y_val)
search.run()
