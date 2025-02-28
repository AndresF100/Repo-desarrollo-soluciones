from pathlib import Path
import pandas as pd
from data_preprocessing.data_loader import load_data



BASE_DIR = Path().resolve().parent.parent
BASE_DIR

data = load_data("clasificacion_siniestros.xlsx")