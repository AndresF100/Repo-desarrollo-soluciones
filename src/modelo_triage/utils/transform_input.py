import sys
import os
import pandas as pd
import joblib

# Asegurar las rutas
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
sys.path.append(base_path)

feature_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", "..","data_preprocessing"))	
sys.path.append(feature_path)

# Importar módulos que el pipeline necesita
from data_preprocessing.data_transformation import transform_data


def prepare_input_data(input_df: pd.DataFrame):
    """
    Aplica transformaciones al DataFrame de entrada utilizando el pipeline pre-entrenado.
    """
    # Evitar la sobreescritura del DataFrame original
    input_df = input_df.copy()

    # Aplicar transformaciones básicas
    transformed_df = transform_data(input_df)

    # Ruta al pipeline entrenado
    pipeline_path = os.path.join(base_path, "data_preprocessing", "trained_pipelines", "transformation_pipeline.pkl")

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"❌ No se encontró el pipeline en: {pipeline_path}")

    # Cargar el pipeline
    try:
        pipeline = joblib.load(pipeline_path)
        print("✅ Pipeline cargado exitosamente.")
    except (ModuleNotFoundError, Exception) as e:
        raise ImportError(f"❌ Error al cargar el pipeline: {e}")

    # Validar que el pipeline se haya cargado correctamente
    if pipeline is None:
        raise RuntimeError("❌ El pipeline no se cargó correctamente. Revisa el proceso de serialización.")

    # Transformar los datos
    try:
        transformed_data = pipeline.transform(transformed_df)
    except Exception as e:
        raise RuntimeError(f"❌ Error al transformar los datos: {e}")

    return transformed_data


# # Prueba
# if __name__ == "__main__":
#     input_file = os.path.join(base_path,"..", "data", "user_input_example", "user_input.csv")

#     if not os.path.exists(input_file):
#         raise FileNotFoundError(f"❌ No se encontró el archivo de entrada: {input_file}")

#     df = pd.read_csv(input_file, sep=';')
#     df_sample = df.iloc[:2].copy()

#     print("📊 Datos de entrada:")
#     print(df_sample)

#     transformed_data = prepare_input_data(df_sample)
#     print("✅ Transformación completada. Resultado:")
#     print(transformed_data)
