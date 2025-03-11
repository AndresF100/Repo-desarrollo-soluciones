import os
import pandas as pd
import mlflow.pyfunc
import pytest

# Ruta base del proyecto
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ruta al modelo empaquetado
MODEL_PATH = os.path.join(BASE_PATH, "model_package")

# Ruta al archivo de entrada
INPUT_FILE = os.path.join(BASE_PATH, "data", "user_input_example", "user_input.csv")

@pytest.fixture
def sample_input():
    """
    Carga un subconjunto aleatorio del CSV de entrada para las pruebas.
    """
    if not os.path.exists(INPUT_FILE):
        pytest.fail(f"No se encontró el archivo de entrada: {INPUT_FILE}")
    
    df = pd.read_csv(INPUT_FILE, sep=';')
    return df.sample(n=3, random_state=42)  # Tomar 3 filas aleatorias para probar

@pytest.fixture(scope="module")
def loaded_model():
    """
    Carga el modelo empaquetado desde el directorio 'model_package'.
    """
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"No se encontró el modelo empaquetado en: {MODEL_PATH}")
    
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    return model

def test_model_prediction(loaded_model, sample_input):
    """
    Verifica que el modelo genera predicciones a partir de los datos de entrada.
    """
    # Validar que el DataFrame no esté vacío
    assert not sample_input.empty, "El DataFrame de entrada está vacío."
    
    # Realizar predicción
    predictions = loaded_model.predict(sample_input)

    # Validar que las predicciones no están vacías
    assert predictions is not None, "Las predicciones no deberían ser None."
    assert len(predictions) == len(sample_input), "La cantidad de predicciones no coincide con la entrada."
    
    print("✅ Predicción exitosa:", predictions)





