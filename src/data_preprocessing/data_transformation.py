import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.joinpath("data", "visual", "data_for_visuals.csv")


def transform_data(df: pd.DataFrame) -> pd.DataFrame:

    # 1. Remapear valores de la columna 'ind_realizando_trabajo_hab_at_igatepmafurat'
    df['ind_realizando_trabajo_hab_at_igatepmafurat'] = df['ind_realizando_trabajo_hab_at_igatepmafurat'].map(
        {"si":"SI",
        "s":"SI",
        "sin informacion":"SIN INFORMACION",
        "no":"NO",
        "n":"NO",
        "1":"SI"})

    logging.info(f"\tSe han remapeado los valores de la columna 'ind_realizando_trabajo_hab_at_igatepmafurat'.")

    # 2. Ajuste de columnas de categoría
    # Identificar columnas que empiezan con "ind" o "id"
    cols_to_str = df.filter(regex='^(ind|id|emp|tipo|seg|centro)').columns


    # Convertir las columnas seleccionadas a tipo str
    df[cols_to_str] = df[cols_to_str].astype(str)

    # permite reutilizar el código en datasets de entrada para predicción
    if 'origen_igdactmlmacalificacionorigen' in df.columns:
        # Variable a predecir como categórica
        df['origen_igdactmlmacalificacionorigen'] = df['origen_igdactmlmacalificacionorigen'].astype(str)

    logging.info(f"\tSe han ajustado las columnas de categoría.")

    # 3. reemplazar valores fuera de s y n 
    columnas_1 = ['dto_igdacmlmasolicitudes', 'pcl_igdacmlmasolicitudes']
    columnas_2 = ['accidente_grave_igatepmafurat', 'riesgo_biologico_igatepmafurat']

    # Aplicar reemplazos en las columnas correspondientes
    df[columnas_1] = df[columnas_1].replace('', 'n')
    df[columnas_2] = df[columnas_2].replace('0', 'n')

    logging.info(f"\tSe han imputado los espacios vacíos o ceros en variables binarias.") 

    # Guardar en datos para visualización
    # df.to_csv(OUTPUT_DIR, index=False)


    # 4. Extraer mes y día del siniestro de la columna 'fecha_siniestro_igdacmlmasolicitudes'
    # Convertir a datetime
    df['fecha_siniestro_igdacmlmasolicitudes'] = pd.to_datetime(df['fecha_siniestro_igdacmlmasolicitudes'])

    df['fecha_siniestro_month'] = df['fecha_siniestro_igdacmlmasolicitudes'].dt.month
    df['fecha_siniestro_day'] = df['fecha_siniestro_igdacmlmasolicitudes'].dt.day

    df.drop(columns=['fecha_siniestro_igdacmlmasolicitudes'], inplace=True)

    logging.info(f"\tSe han extraído las variables temporales 'mes' y 'día' del siniestro.")


    # 5. Capturar periodicidad en variables temporales

    # Ciclo fijo de 12 meses (1-12)
    df["fecha_siniestro_month_sin"] = np.sin(2 * np.pi * (df["fecha_siniestro_month"] - 1) / 12)
    df["fecha_siniestro_month_cos"] = np.cos(2 * np.pi * (df["fecha_siniestro_month"] - 1) / 12)

    # Ciclo variable de días según el mes (1-31)
    dias_por_mes = np.array([
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    ])

    # Mapear días máximos según el mes
    dias_maximos = df["fecha_siniestro_month"].map(lambda x: dias_por_mes[x - 1])

    # Calcular seno y coseno ajustado al mes
    df["fecha_siniestro_day_sin"] = np.sin(2 * np.pi * (df["fecha_siniestro_day"] - 1) / dias_maximos)
    df["fecha_siniestro_day_cos"] = np.cos(2 * np.pi * (df["fecha_siniestro_day"] - 1) / dias_maximos)

    # Ciclo fijo de 24 horas (0-23)
    df["hora_siniestro_sin"] = np.sin(2 * np.pi * df["hora_at_igatepmafurat"] / 24)
    df["hora_siniestro_cos"] = np.cos(2 * np.pi * df["hora_at_igatepmafurat"] / 24)

    
    # Ciclo fijo de 24 horas (0-23)
    df["hora_previo_sin"] = np.sin(2 * np.pi * df["horas_previo_at_igatepmafurat"] / 24)
    df["hora_previo_cos"] = np.cos(2 * np.pi * df["horas_previo_at_igatepmafurat"] / 24)


    # Remover columnas originales
    df.drop(columns=["fecha_siniestro_month", "fecha_siniestro_day", "hora_at_igatepmafurat","horas_previo_at_igatepmafurat"], inplace=True)
    
    logging.info(f"\tSe han capturado las periodicidades en las variables temporales con seno y coseno.") 


    return df

