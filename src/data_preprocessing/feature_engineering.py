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

    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.mappings = {}
        self.high_cardinality_cols = []

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if X[col].nunique() / len(X) > self.threshold:
                self.high_cardinality_cols.append(col)
                self.mappings[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.high_cardinality_cols:
            X[col + '_freq'] = X[col].map(self.mappings[col]).fillna(0)
        return X.drop(columns=self.high_cardinality_cols)

def detect_column_types(df, high_cardinality_threshold=0.05):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() / len(df) > high_cardinality_threshold]
    categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]
    
    return numerical_cols, categorical_cols, high_cardinality_cols

def create_feature_engineering_pipeline(df):
    numerical_cols, categorical_cols, high_cardinality_cols = detect_column_types(df)
    
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
        ('high_cardinality', HighCardinalityEncoder()),
        ('features', feature_engineering)
    ])

    return preprocessing_pipeline

def split_data(df, target_column, test_size=0.2, val_size=0.1, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size / (1 - test_size), random_state=random_state)
    
    return train_df, val_df, test_df

def transform_and_split_data(df, target_column='origen_igdactmlmacalificacionorigen'):
    # Divide los datos
    train_df, val_df, test_df = split_data(df, target_column)

    # Separa las features (X) y la variable objetivo (y)
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    # Crea el pipeline de transformación
    pipeline = create_feature_engineering_pipeline(X_train)

    # Ajusta el pipeline solo con X_train y aplica a los otros conjuntos
    X_train_transformed = pipeline.fit_transform(X_train)
    X_val_transformed = pipeline.transform(X_val)
    X_test_transformed = pipeline.transform(X_test)

    return X_train_transformed, y_train, X_val_transformed, y_val, X_test_transformed, y_test