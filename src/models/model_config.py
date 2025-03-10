from itertools import product

HYPERPARAM_GRID = {
    "RandomForest": {
        "n_estimators": [1,2,3,4, 6, 8],          # Más árboles para capturar clases raras
        "max_depth": [8,9,10],                # Mayor profundidad para patrones complejos
        "min_samples_split": [2, 4],         # Controla la división mínima
        "min_samples_leaf": [1,],          # Garantiza no perder clases raras
        "class_weight": ["balanced"],           # Ajusta automáticamente los pesos
        "max_features": ["sqrt", None], # Controla el número de variables por split
        "criterion": ["gini", "entropy"],       # Mide la calidad de la división
        "max_samples": [0.7, 0.9]         # Submuestreo para más diversidad
             
             },
    "XGBoost": {
        "n_estimators": [100, 300],
        "learning_rate": [0.01, 0.1],
        "max_depth": [4, 8]
    }
}

def get_param_combinations(model_name):
    grid = HYPERPARAM_GRID[model_name]
    return [dict(zip(grid.keys(), values)) for values in product(*grid.values())]
