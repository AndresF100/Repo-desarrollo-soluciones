# import pandas as pd
# from model.predict import make_prediction

# sample_input_data = pd.read_csv("~/test/bankchurn_test.csv")
# result = make_prediction(input_data=sample_input_data)
# print(result)
# from modelo_triage.loader import get_model

import os
import sys
import pandas as pd

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
sys.path.append(BASE_PATH)

# Ruta al archivo de entrada
INPUT_FILE = os.path.join(BASE_PATH, "data", "user_input_example", "user_input.csv")


df = pd.read_csv(INPUT_FILE, sep=';')
data= df.sample(n=3, random_state=42)


from modelo_triage.loader import get_model

modelo = get_model()

prediction=modelo.predict(data)
print(prediction)


