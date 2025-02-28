from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_data(filename: str, folder="raw"):
    """
    Carga un archivo CSV desde la carpeta data/.
    """
    file_path = BASE_DIR / "data" / folder / filename  

    if not file_path.exists():
        raise FileNotFoundError(f"El archivo {file_path} no existe.")

    logging.info(f"Cargando datos desde: {file_path}")
    return pd.read_csv(file_path, sep=";", encoding='utf-8',low_memory=False)


