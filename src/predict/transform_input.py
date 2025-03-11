import sys
import os
import pandas as pd
import joblib

# Asegurar las rutas
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_path)


features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..","src","data_preprocessing"))
sys.path.append(features_path)

# Importar módulos que el pipeline necesita
from src.data_preprocessing.data_transformation import transform_data


def prepare_input_data(input_df: pd.DataFrame):
    # Evitar la sobreescritura
    input_df = input_df.copy()

    # Aplicar transformaciones básicas
    transformed_df = transform_data(input_df)

    # Ruta al pipeline entrenado
    pipeline_path = os.path.join(base_path, "src", "data_preprocessing", "trained_pipelines", "transformation_pipeline.pkl")

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"No se encontró el pipeline en {pipeline_path}")

    # Forzar la importación del módulo requerido
    print(f"🔍 Cargando el pipeline desde: {pipeline_path}")
    try:
        pipeline = joblib.load(pipeline_path)
    except ModuleNotFoundError as e:
        print(f"❌ Error al cargar el pipeline: {e}")

    print("✅ Pipeline cargado exitosamente.")

    # Transformar los datos
    transformed_data = pipeline.transform(transformed_df)

    return transformed_data

# # Prueba
# if __name__ == "__main__":
#     input_file = os.path.join(base_path, "data", "user_input_example", "user_input.csv")
#     df = pd.read_csv(input_file, sep=';')
#     df_sample = df.iloc[:2].copy()

#     transformed_data = prepare_input_data(df_sample)
#     print("✅ Transformación completada.")
