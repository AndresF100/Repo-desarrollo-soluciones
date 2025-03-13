import mlflow
import os
import pandas as pd
import sys

# 📂 Ruta absoluta al modelo empaquetado
base_path = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(base_path,"..", "model_package")

upper_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
sys.path.append(upper_path)


if not os.path.exists(OUTPUT_DIR):
    raise FileNotFoundError(f"❌ No se encontró el modelo empaquetado en: {OUTPUT_DIR}")

def get_model():
    """
    Carga el modelo empaquetado y lo devuelve.
    """
    # Cargar el modelo empaquetado
    packaged_model = mlflow.pyfunc.load_model(OUTPUT_DIR)
    print(f"✅ Modelo cargado desde: {OUTPUT_DIR}")
    return packaged_model
    

# # ✅ Ejemplo de prueba
# if __name__ == "__main__":
#     # 📊 Ruta al archivo de entrada
#     input_file = os.path.join(base_path,"..", "..", "data", "user_input_example", "user_input.csv")

#     if not os.path.exists(input_file):
#         raise FileNotFoundError(f"❌ No se encontró el archivo de entrada: {input_file}")

#     # Cargar el DataFrame de ejemplo
#     df = pd.read_csv(input_file, sep=";")
#     print("📊 Datos de entrada:")
#     print(df.head())

#     # Realizar predicciones
#     modelo = get_model()
#     predictions = modelo.predict(df)
#     print("🔮 Predicciones:")
#     print(predictions)
