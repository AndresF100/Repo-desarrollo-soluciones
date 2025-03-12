import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import joblib

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


def balance_classes(X_train, y_train):
    """
    Balancea las clases de un conjunto de datos.
    Se aplica un undersampling a la clase mayoritaria y un oversampling a las clases minoritarias.
    de la siguiente manera:
    1. Se reduce la clase mayoritaria al tamaño de la segunda clase mayoritaria.
    2. Se aumenta el tamaño de las clases minoritarias al 25% del tamaño de la segunda clase mayoritaria.

    """
    class_counts = np.bincount(y_train)
    sorted_counts = np.sort(class_counts)[::-1]

    if len(sorted_counts) < 4:
        raise ValueError("Se esperaban al menos 4 clases para aplicar este balanceo.")

    second_majority_count = sorted_counts[1]
    minority_target_size = int(second_majority_count * 0.25)

    # 1. Reducir la clase mayoritaria al tamaño de la segunda mayoritaria
    target_counts = {
        cls: second_majority_count if count == sorted_counts[0] else count
        for cls, count in enumerate(class_counts)
    }

    under_sampler = RandomUnderSampler(sampling_strategy=target_counts, random_state=42)
    X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)

    # 2. Recalcular las clases después del undersampling
    new_class_counts = np.bincount(y_train_resampled)
    second_majority_count = sorted(new_class_counts)[-2]

    # 3. Ajustar SMOTE dinámicamente
    smote_strategy = {}
    for cls, count in enumerate(new_class_counts):
        if count < minority_target_size and count >= 2:
            # Asegurar que k_neighbors <= n_samples - 1
            k_neighbors = min(5, count - 1)
            smote_strategy[cls] = minority_target_size

    if smote_strategy:
        smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=k_neighbors, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_resampled, y_train_resampled)

    return X_train_resampled, pd.Series(y_train_resampled)



def transform_and_split_data(df, target_column='origen_igdactmlmacalificacionorigen'):
    train_df, val_df, test_df = split_data(df, target_column)

    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    # Codificar y si es categórico
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_val = label_encoder.transform(y_val)
        y_test = label_encoder.transform(y_test)

    pipeline = create_feature_engineering_pipeline(X_train)

    X_train_transformed = pipeline.fit_transform(X_train)

    # Guardar el pipeline para usarlo en la API
    joblib.dump(pipeline, './trained_pipelines/transformation_pipeline.pkl')

    X_val_transformed = pipeline.transform(X_val)
    X_test_transformed = pipeline.transform(X_test)

    X_train_transformed, X_val_transformed, X_test_transformed = check_dimensions(X_train_transformed, X_val_transformed, X_test_transformed)

    # Balanceo mixto (undersampling + oversampling)
    X_train_transformed, y_train = balance_classes(X_train_transformed, y_train)

    return X_train_transformed, pd.Series(y_train), X_val_transformed, pd.Series(y_val), X_test_transformed, pd.Series(y_test)