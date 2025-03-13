"""
Adapter module for translating simple API inputs to the format required by the model.
"""

import pandas as pd
import numpy as np

# Field mappings from API inputs to model columns
FIELD_MAPPINGS = {
    'parte_cuerpo': 'id_parte_cuerpo_igatepmafurat',
    'municipio': 'id_municipio_at_igatepmafurat',  
    'jornada_trabajo': 'ind_tipo_jornada_at_igatepmafurat',
    'realizando_trabajo': 'ind_realizando_trabajo_hab_at_igatepmafurat',
    'descripcion': 'descripcion_at_igatepmafurat'
}

# Default values for required fields based on most frequent values in training data
# Making sure numeric fields are properly typed
DEFAULT_VALUES = {
    'dto_igdacmlmasolicitudes': 's',
    'pcl_igdacmlmasolicitudes': 's', 
    'tipo_siniestro_igdacmlmasolicitudes': 0,  # Numeric
    'fecha_siniestro_igdacmlmasolicitudes': '2009-03-05 00:00:00+00:00',
    'hora_at_igatepmafurat': 13,  # Numeric - this is the field causing the error
    'horas_previo_at_igatepmafurat': 5,  # Numeric
    'ind_sitio_ocurrencia_igatepmafurat': 2,  # Numeric
    'id_tipo_lesion_igatepmafurat': 55,  # Numeric
    'id_agente_at_igatepmafurat': 5,  # Numeric
    'id_mecanismo_at_igatepmafurat': 1,  # Numeric
    'ind_testigo_at_igatepmafurat': 1,  # Numeric
    'id_medio_recepcion_igatepmafurat': 5,  # Numeric
    'centro_trabajo_igual_igatepmafurat': 1,  # Numeric
    'accidente_grave_igatepmafurat': 'n',
    'riesgo_biologico_igatepmafurat': 'n',
    'id_sitio_ocurrencia_igatepmafurat': 1  # Numeric
}

# Data types for each column
COLUMN_TYPES = {
    'tipo_siniestro_igdacmlmasolicitudes': 'int64',
    'hora_at_igatepmafurat': 'int64', 
    'horas_previo_at_igatepmafurat': 'int64',
    'ind_sitio_ocurrencia_igatepmafurat': 'int64',
    'id_tipo_lesion_igatepmafurat': 'int64',
    'id_agente_at_igatepmafurat': 'int64',
    'id_mecanismo_at_igatepmafurat': 'int64',
    'ind_testigo_at_igatepmafurat': 'int64',
    'id_medio_recepcion_igatepmafurat': 'int64',
    'centro_trabajo_igual_igatepmafurat': 'int64',
    'id_sitio_ocurrencia_igatepmafurat': 'int64',
    'id_municipio_at_igatepmafurat': 'int64',
    'id_parte_cuerpo_igatepmafurat': 'int64',
    'ind_tipo_jornada_at_igatepmafurat': 'int64',
}

# Value mappings for specific fields
VALUE_MAPPINGS = {
    'jornada_trabajo': {
        'SI': 's',
        'NO': 'n',
        'SIN INFORMACION': 'sin informacion'
    }
}

def create_model_input(api_input: dict) -> pd.DataFrame:
    """
    Creates a DataFrame with all required columns for the model, 
    using the API input and default values.
    
    Args:
        api_input: Dictionary with API request fields
        
    Returns:
        pd.DataFrame: A DataFrame formatted for model consumption
    """
    # Create a dictionary with all default values
    model_input = DEFAULT_VALUES.copy()
    
    # Map API input fields to model fields
    for api_field, model_field in FIELD_MAPPINGS.items():
        if api_field in api_input and api_input[api_field] is not None:
            # Handle special case for descripcion which might be empty
            if api_field == 'descripcion' and api_input[api_field] == "":
                model_input[model_field] = "no description provided"
            # Handle special case for jornada_trabajo mapping
            elif api_field == 'jornada_trabajo' and api_input[api_field] in VALUE_MAPPINGS['jornada_trabajo']:
                model_input[model_field] = VALUE_MAPPINGS['jornada_trabajo'][api_input[api_field]]
            # Handle numeric fields
            elif model_field in COLUMN_TYPES and COLUMN_TYPES[model_field] == 'int64':
                try:
                    model_input[model_field] = int(api_input[api_field])
                except (ValueError, TypeError):
                    # If conversion fails, keep the original value and log a warning
                    print(f"Warning: Could not convert {api_field}='{api_input[api_field]}' to integer")
                    model_input[model_field] = api_input[api_field]
            else:
                model_input[model_field] = api_input[api_field]
    
    # Create a pandas DataFrame (single row)
    df = pd.DataFrame([model_input])
    
    # Ensure proper data types for all columns
    for column, dtype in COLUMN_TYPES.items():
        if column in df.columns:
            try:
                df[column] = df[column].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not convert column {column} to {dtype}: {e}")
    
    return df