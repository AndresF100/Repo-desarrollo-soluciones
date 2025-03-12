import os
import pandas as pd
import pytest
import sys

# Ruta base del proyecto


MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(MODEL_PATH)


# Importar la función de predicción
from src.loader import get_model

modelo = get_model()

# Ruta al archivo de entrada
INPUT_FILE = os.path.join(BASE_PATH, "data", "user_input_example", "user_input.csv")

@pytest.fixture
def sample_input():
    """
    Carga un subconjunto aleatorio del CSV de entrada para las pruebas.
    """
    if not os.path.exists(INPUT_FILE):
        pytest.fail(f"❌ No se encontró el archivo de entrada: {INPUT_FILE}")
    
    df = pd.read_csv(INPUT_FILE, sep=';')
    return df.sample(n=3, random_state=42)  # Tomar 3 filas aleatorias para probar

def test_model_prediction(sample_input):
    """
    Verifica que el modelo genera predicciones a partir de los datos de entrada.
    """
    # Validar que el DataFrame no esté vacío
    assert not sample_input.empty, "❌ El DataFrame de entrada está vacío."
    
    # Realizar predicciones
    modelo = get_model()
    predictions = modelo.predict(sample_input)

    # Validar que las predicciones no están vacías
    assert predictions is not None, "❌ Las predicciones no deberían ser None."
    assert len(predictions) == len(sample_input), "❌ La cantidad de predicciones no coincide con la entrada."
    
    print("✅ Predicción exitosa:", predictions)


# # Realizar predicciones
# modelo = get_model()

# if not os.path.exists(INPUT_FILE):
#     pytest.fail(f"❌ No se encontró el archivo de entrada: {INPUT_FILE}")

# df = pd.read_csv(INPUT_FILE, sep=';')
# data= df.sample(n=3, random_state=42)

# predictions = modelo.predict(data)

# print("🔮 Predicciones:")
# print(predictions)