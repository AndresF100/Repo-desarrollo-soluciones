import logging
import pandas as pd
import numpy as np
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Campos que son identificación de registros (Ids)
ids_fields = [
    "ID_FURAT_FUREP_IGDACMLMASOLICITUDES",
    "emp_Id_IGDACMLMASOLICITUDES",
    "ID_SOLICITUD_IGDACMLMASOLICITUDES",
    "ID_TIPO_DOC_EMP_IGDACMLMASOLICITUDES",
    "ID_SINIESTRO_IGATEPMAFURAT",
    "ID_OCUPACION_AT_IGATEPMAFURAT",
    "IND_ZONA_IGATEPMAFURAT",
    "ID_CALIFICACION_DTO_IGDACTMLMACALIFICACIONORIGEN",
    "ID_SOLICITUD_IGDACTMLMACALIFICACIONORIGEN",
    "id_furat_igatepmafurat"
]


# Campos que desde el contexto de negocio no aportan información
non_informative_fields = [
    "seg_idPonderado_IGDACMLMASOLICITUDES",
    "MONTO_RESERVA_IGATEPMAFURAT",
    "ESTADO_RESERVA_IGATEPMAFURAT",
    "ID_SOLICITUD_IGACCTMIMVDIAGNOSTICOS",
    "CONS_DIAG_IGACCTMIMVDIAGNOSTICOS",
    "NOMBRE_DIAGNOSTICO_IGACCTMIMVDIAGNOSTICOS",
    "ID_DX_IGACCTMIMVDIAGNOSTICOS",
    "TIPO_CALIFICADOR_IGACCTMIMVDIAGNOSTICOS",
    "TIPO_CALIFICACION_IGACCTMIMVDIAGNOSTICOS",
    "FECHA_MODIFICACION_AUD_IGACCTMIMVDIAGNOSTICOS",
    "FECHA_DICTAMEN_IGDACTMLMACALIFICACIONORIGEN",
    "FECHA_ESTRUCTURACION_IGDACTMLMACALIFICACIONORIGEN",
    "FECHA_MODIFICACION_AUD_IGDACTMLMACALIFICACIONORIGEN",
    "DIAS_INCAPACIDAD_IGATEPMAFURAT",
    "FECHA_MODIFICACION_AUD_IGATEPMAFURAT",
    "fecha_diligenciamiento_igatepmafurat",
    "fecha_radicacion_igatepmafurat",
    "fecha_solicitud_igdacmlmasolicitudes",
    "fecha_modificacion_aud_igdacmlmasolicitudes"
]

# Variables correlacionadas
correlated_fields = [
    "ID_DEPARTAMENTO_AT_IGATEPMAFURAT"  # Se elimina
]

# Otras variables a descartar
other_fields = [
    "ind_tipo_at_igatepmafurat", # es constante tras limpieza
    "ind_muerte_igatepmafurat", # es constante tras limpieza
    "otro_sitio_ocurrencia_igatepmafurat", # texto libre, puede ser útil pero requiere trabajo adicional
    "otro_tipo_lesion_igatepmafurat", # texto libre, puede ser útil pero requiere trabajo adicional
    "otro_mecanismo_accidente_igatepmafurat", # texto libre, puede ser útil pero requiere trabajo adicional
    "nombre_ocupacion_igatepmafurat", # texto libre, puede ser útil pero requiere trabajo adicional
    "minuto_at_igatepmafurat", # tenemos el agrupador de hora
    "minutos_previo_at_igatepmafurat", # tenemos el agrupador de hora
    "oficio_solicitud_igatepmafurat", # en su mayoría son No o vacíos
]


# Lista de columnas no útiles para el pronóstico
COLUMNS_TO_DROP = ids_fields + non_informative_fields + correlated_fields + other_fields
    


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    original_shape = df.shape
    logging.info(f'\tIniciando limpieza: {original_shape} registros.')

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

    # 4. Eliminar columnas con más del 50% de valores nulos
    threshold = 0.5
    null_cols = df.columns[df.isnull().mean() > threshold]
    if len(null_cols) > 0:
        df.drop(columns=null_cols, inplace=True)
        logging.info(f'\tColumnas eliminadas por alto porcentaje de nulos (>50%): {list(null_cols)}')

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
            logging.warning(f'\tError al convertir columna {col} a datetime: {e}')

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
    logging.info(f'\tTotal registros eliminados por outliers: {base_rows - actual_rows}.')

    # 8. Estandarizar texto
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()

    # 9. Limpiar columna de texto descripción
    text_col = 'descripcion_at_igatepmafurat'
    df[text_col] = df[text_col].str.replace(r'\W', ' ', regex=True).str.replace(r'\s+', ' ', regex=True)
     

    # 10. Registrar cambios finales
    df.dropna(inplace=True)
    final_shape = df.shape
    logging.info(f'\tLimpieza completada: {final_shape} registros (cambio de {original_shape} a {final_shape}).')

    return df