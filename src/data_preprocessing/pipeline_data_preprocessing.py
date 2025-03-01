import logging
from data_loader import load_data
from data_cleaning import clean_data
from data_transformation import transform_data
from feature_engineering import transform_and_split_data
import pandas as pd
import scipy.sparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_DIR = Path().resolve().parent.parent.joinpath("data", "processed")

def save_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir=OUTPUT_DIR):
    # Guarda las variables objetivo (y) como CSV
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    # Guarda los features (X) como Parquet o CSV
    if isinstance(X_train, scipy.sparse.spmatrix):
        scipy.sparse.save_npz(f"{output_dir}/X_train.npz", X_train)
        scipy.sparse.save_npz(f"{output_dir}/X_val.npz", X_val)
        scipy.sparse.save_npz(f"{output_dir}/X_test.npz", X_test)
    else:
        pd.DataFrame(X_train).to_parquet(f"{output_dir}/X_train.parquet")
        pd.DataFrame(X_val).to_parquet(f"{output_dir}/X_val.parquet")
        pd.DataFrame(X_test).to_parquet(f"{output_dir}/X_test.parquet")

    logging.info(f"✅ Datos guardados en {output_dir}")


# def run_data_preprocessing_pipeline()

logging.info("Iniciando pipeline de preprocesamiento...")

logging.info(f"📊 1. Carga de datos.")
data = load_data("clasificacion_siniestros.csv")
logging.info(f"✅ Datos cargados: {data.shape[0]} filas y {data.shape[1]} columnas.")

logging.info(f"🧹 2. Limpieza de datos.")
data = clean_data(data)
logging.info(f"✅ Proceso de limpieza finalizado.")

logging.info(f"🔄 3. Transformación de datos.")
data = transform_data(data)
logging.info(f"✅ Proceso de transformación finalizado.")

logging.info(f"🔧 4. Ingeniería de características y partición de datos")
x_train, y_train, x_val, y_val, x_test, y_test = transform_and_split_data(data.copy())
logging.info(f"✅ Proceso de ingeniería de características y partición de datos finalizado.")

logging.info(f"📦 5. Guardando datos procesados.")
save_data(x_train, x_val, x_test, y_train, y_val, y_test)

