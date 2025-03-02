import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class HighCardinalityEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, high_cardinality_cols):
        self.high_cardinality_cols = high_cardinality_cols
        self.mappings = {}

    def fit(self, X, y=None):
        for col in self.high_cardinality_cols:
            self.mappings[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.high_cardinality_cols:
            X[col + '_freq'] = X[col].map(lambda x: self.mappings[col].get(x, 0))
        return X.drop(columns=self.high_cardinality_cols)

def detect_column_types(df, target_col, high_cardinality_threshold=20):
    text_col = "descripcion_at_igatepmafurat"

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    exclude_cols = {text_col, target_col}
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > high_cardinality_threshold]

    categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]

    return numerical_cols, categorical_cols, high_cardinality_cols, text_col

def create_feature_engineering_pipeline(df):
    numerical_cols, categorical_cols, high_cardinality_cols, text_col = detect_column_types(df, target_col='origen_igdactmlmacalificacionorigen')

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    high_cardinality_transformer = Pipeline(steps=[
        ('high_cardinality', HighCardinalityEncoder(high_cardinality_cols))
    ])

    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=200))
    ])

    feature_engineering = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('high_card', high_cardinality_transformer, high_cardinality_cols),
            ('text', text_transformer, text_col),
        ],
        remainder='drop'
    )

    preprocessing_pipeline = Pipeline(steps=[
        ('features', feature_engineering)
    ])

    return preprocessing_pipeline

def split_data(df, target_column, test_size=0.2, val_size=0.1, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size / (1 - test_size), random_state=random_state)
    return train_df, val_df, test_df

def check_dimensions(X_train, X_val, X_test):
    min_features = min(X_train.shape[1], X_val.shape[1], X_test.shape[1])
    return X_train[:, :min_features], X_val[:, :min_features], X_test[:, :min_features]

def transform_and_split_data(df, target_column='origen_igdactmlmacalificacionorigen'):
    train_df, val_df, test_df = split_data(df, target_column)

    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    pipeline = create_feature_engineering_pipeline(X_train)

    X_train_transformed = pipeline.fit_transform(X_train)
    X_val_transformed = pipeline.transform(X_val)
    X_test_transformed = pipeline.transform(X_test)

    X_train_transformed, X_val_transformed, X_test_transformed = check_dimensions(X_train_transformed, X_val_transformed, X_test_transformed)

    return X_train_transformed, y_train, X_val_transformed, y_val, X_test_transformed, y_test
