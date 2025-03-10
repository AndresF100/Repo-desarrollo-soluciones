from itertools import product

HYPERPARAM_GRID = {
    "RandomForest": {
        "n_estimators": [100, 200, 500],         # Más árboles para capturar mejor las clases minoritarias
        "max_depth": [10, 20, 30],               # Mayor profundidad para encontrar patrones complejos
        "min_samples_split": [2, 5],             # Controla la división mínima
        "min_samples_leaf": [1, 3],              # Garantiza que no se pierdan clases raras
        "class_weight": ["balanced"]             # Ajusta automáticamente los pesos según la frecuencia
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
