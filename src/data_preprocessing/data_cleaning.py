import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Lista de columnas no útiles para el pronóstico
COLUMNS_TO_DROP = [
    "ID_FURAT_FUREP_IGDACMLMASOLICITUDES", "ID_SOLICITUD_IGDACMLMASOLICITUDES", "FECHA_SOLICITUD_IGDACMLMASOLICITUDES",
    "FECHA_MODIFICACION_AUD_IGDACMLMASOLICITUDES", "DTO_IGDACMLMASOLICITUDES", "PCL_IGDACMLMASOLICITUDES",
    "ID_FURAT_IGATEPMAFURAT", "ID_SINIESTRO_IGATEPMAFURAT", "MINUTO_AT_IGATEPMAFURAT", "NOMBRE_OCUPACION_IGATEPMAFURAT",
    "HORAS_PREVIO_AT_IGATEPMAFURAT", "MINUTOS_PREVIO_AT_IGATEPMAFURAT", "FECHA_MUERTE_IGATEPMAFURAT",
    "IND_TESTIGO_AT_IGATEPMAFURAT", "FECHA_DILIGENCIAMIENTO_IGATEPMAFURAT", "FECHA_RADICACION_IGATEPMAFURAT",
    "ID_MEDIO_RECEPCION_IGATEPMAFURAT", "OFICIO_SOLICITUD_IGATEPMAFURAT", "CENTRO_TRABAJO_IGUAL_IGATEPMAFURAT",
    "ACCIDENTE_GRAVE_IGATEPMAFURAT", "RIESGO_BIOLOGICO_IGATEPMAFURAT", "ID_SITIO_OCURRENCIA_IGATEPMAFURAT",
    "MONTO_RESERVA_IGATEPMAFURAT", "DIAS_INCAPACIDAD_IGATEPMAFURAT", "MUERTE_POSTERIOR_IGATEPMAFURAT",
    "ESTADO_RESERVA_IGATEPMAFURAT", "FECHA_AVISO_MUERTE_IGATEPMAFURAT", "ID_SOLICITUD_IGACCTMIMVDIAGNOSTICOS",
    "CONS_DIAG_IGACCTMIMVDIAGNOSTICOS", "NOMBRE_DIAGNOSTICO_IGACCTMIMVDIAGNOSTICOS",
    "TIPO_CALIFICADOR_IGACCTMIMVDIAGNOSTICOS", "TIPO_CALIFICACION_IGACCTMIMVDIAGNOSTICOS",
    "ID_CALIFICACION_DTO_IGDACTMLMACALIFICACIONORIGEN", "ID_SOLICITUD_IGDACTMLMACALIFICACIONORIGEN",
    "ind_tipo_at_igatepmafurat", # es constante tras limpieza
    "ind_muerte_igatepmafurat", # es constante tras limpieza
    "otro_sitio_ocurrencia_igatepmafurat", # texto libre, puede ser útil pero requiere trabajo adicional
    "otro_tipo_lesion_igatepmafurat", # texto libre, puede ser útil pero requiere trabajo adicional
    "otro_mecanismo_accidente_igatepmafurat", # texto libre, puede ser útil pero requiere trabajo adicional
    "fecha_modificacion_aud_igatepmafurat", # fecha sin explicación en diccionario
    "fecha_dictamen_igdactmlmacalificacionorigen", # fecha sin explicación en diccionario
    "fecha_estructuracion_igdactmlmacalificacionorigen", # fecha sin explicación en diccionario
    "fecha_modificacion_aud_igdactmlmacalificacionorigen", # fecha sin explicación en diccionario


]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    original_shape = df.shape
    logging.info(f'Iniciando limpieza: {original_shape} registros.')

    # Eliminar primer columna (id de la fila)
    df.drop(df.columns[0], axis=1, inplace=True)


    # 1. Estandarizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

    # 2. Eliminar duplicados
    df.drop_duplicates(inplace=True)

    # 3. Eliminar columnas no útiles para el pronóstico
    cols_to_remove = [col.lower() for col in COLUMNS_TO_DROP]
    cols_to_drop = [col for col in df.columns if col in cols_to_remove]

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logging.info(f'Columnas eliminadas por irrelevancia: {cols_to_drop}')

    # 4. Eliminar columnas con más del 50% de valores nulos
    threshold = 0.5
    null_cols = df.columns[df.isnull().mean() > threshold]
    if len(null_cols) > 0:
        df.drop(columns=null_cols, inplace=True)
        logging.info(f'Columnas eliminadas por alto porcentaje de nulos (>50%): {list(null_cols)}')

    # 5. Imputar valores nulos numéricos con la mediana
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # 6. Corregir tipos de datos fecha
    date_format = "%Y-%m-%d %H:%M:%S%z"
    for col in df.filter(regex='^fecha').columns:
        try:
            df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
        except Exception as e:
            logging.warning(f'Error al convertir columna {col} a datetime: {e}')

    # 7. Eliminar outliers (método IQR) en columnas numéricas (excluye las que empiezan por "origen")
    base_rows = df.shape[0]
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Excluir columnas que empiezan por "origen"
    numeric_cols = [col for col in numeric_cols if not col.startswith('origen')]

    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.10), df[col].quantile(0.90)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        if outliers > 0:
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    actual_rows = df.shape[0]
    logging.info(f'Total registros eliminados por outliers: {base_rows - actual_rows}.')

    # 8. Estandarizar texto
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()

    # 9. Registrar cambios finales
    final_shape = df.shape
    logging.info(f'Limpieza completada: {final_shape} registros (cambio de {original_shape} a {final_shape}).')

    return df