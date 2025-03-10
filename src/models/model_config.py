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
    # Número de árboles
    "n_estimators": [100, 300, 500],  
    
    # Tasa de aprendizaje (trade-off entre velocidad y precisión)
    "learning_rate": [0.01, 0.05, 0.1],  
    
    # Profundidad máxima de los árboles (controla sobreajuste)
    "max_depth": [4, 6, 8, 12],  
    
    # Controla la regularización L1 y L2 (reduce sobreajuste)
    "reg_alpha": [0, 0.1, 1],        # Regularización L1
    "reg_lambda": [1, 5, 10],        # Regularización L2
    
    # Ajusta el peso de las clases (para balancear el dataset)
    "scale_pos_weight": ["balanced", 5, 10, 20],  
    
    # Porcentaje de muestras utilizadas para cada árbol (submuestreo fila)
    "subsample": [0.7, 0.8, 1.0],  
    
    # Porcentaje de características usadas en cada árbol (submuestreo columna)
    "colsample_bytree": [0.5, 0.7, 1.0],  
    
    # Mínimo de muestras por hoja (controla ramas en clases minoritarias)
    "min_child_weight": [1, 5, 10],  
    
    # Paso máximo del estimador (mejora estabilidad en datasets desbalanceados)
    "max_delta_step": [1, 5, 10]
},
"LightGBM": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 8, 12],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "class_weight": ["balanced"]
    },
    "CatBoost": {
        "iterations": [500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 8, 12],
        "l2_leaf_reg": [3, 10],
        "scale_pos_weight": [1, 5, 10]
    },
    "MLP": {
        "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "max_iter": [300, 500],
        "solver": ["adam"],
    }

}

def get_param_combinations(model_name):
    grid = HYPERPARAM_GRID[model_name]
    return [dict(zip(grid.keys(), values)) for values in product(*grid.values())]
