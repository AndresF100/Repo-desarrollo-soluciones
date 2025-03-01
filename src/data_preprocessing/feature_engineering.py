import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class HighCardinalityEncoder(BaseEstimator, TransformerMixin):
    """Codificación por frecuencia para columnas de alta cardinalidad."""

    def __init__(self, cols):
        self.cols = cols
        self.mappings = {}

    def fit(self, X, y=None):
        for col in self.cols:
            self.mappings[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col + '_freq'] = X[col].map(self.mappings[col]).fillna(0)
        return X.drop(columns=self.cols)

def create_feature_engineering_pipeline():
    categorical_cols = [
        'seg_idponderado_igdacmlmasolicitudes',
        'tipo_siniestro_igdacmlmasolicitudes', 'id_tipo_doc_emp_igdacmlmasolicitudes',
        'ind_tipo_jornada_at_igatepmafurat', 'ind_realizando_trabajo_hab_at_igatepmafurat',
        'ind_zona_igatepmafurat', 'ind_sitio_ocurrencia_igatepmafurat',
        'id_tipo_lesion_igatepmafurat', 'id_parte_cuerpo_igatepmafurat',
        'id_agente_at_igatepmafurat', 'id_mecanismo_at_igatepmafurat'
    ]

    numerical_cols = [
        'fecha_siniestro_month_sin', 'fecha_siniestro_month_cos',
        'fecha_siniestro_day_sin', 'fecha_siniestro_day_cos',
        'hora_siniestro_sin', 'hora_siniestro_cos'
    ]

    high_cardinality_cols = [
        'id_ocupacion_at_igatepmafurat', 'id_departamento_at_igatepmafurat',
        'id_municipio_at_igatepmafurat', 'emp_id_igdacmlmasolicitudes'
    ]

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    feature_engineering = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='drop'
    )

    preprocessing_pipeline = Pipeline(steps=[
        ('high_cardinality', HighCardinalityEncoder(high_cardinality_cols)),
        ('features', feature_engineering)
    ])

    return preprocessing_pipeline

def split_data(df, target_column, test_size=0.2, val_size=0.1, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size / (1 - test_size), random_state=random_state)
    
    return train_df, val_df, test_df

def transform_data(df):
    pipeline = create_feature_engineering_pipeline()
    return pipeline.fit_transform(df)

def transform_and_split_data(df):

    target_column = 'origen_igdactmlmacalificacionorigen'

    train_df, val_df, test_df = split_data(df, target_column)

    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    transformed_train = transform_data(X_train)
    transformed_val = transform_data(X_val)
    transformed_test = transform_data(X_test)

    return transformed_train, y_train, transformed_val, y_val, transformed_test, y_test
