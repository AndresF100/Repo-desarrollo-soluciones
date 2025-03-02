from itertools import product

HYPERPARAM_GRID = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5]
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