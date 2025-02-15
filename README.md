
# Clasificación de Siniestros
Repositorio para el proyecto de clasificación de siniestros dentro de la asignatura Proyecto Desarrollo de soluciones de la Universidad de Los Andes, sigue las prácticas de MLOps para la gestión de datos, entrenamiento de modelos y despliegue en producción. La estructura está diseñada para garantizar trazabilidad, modularidad y escalabilidad en proyectos de Machine Learning.


## 👥 Equipo de Trabajo


A continuación, se listan los miembros del equipo junto con sus usuarios de GitHub:

| Nombre   | Usuario de GitHub  |
|----------|-------------------|
| Alyona Ivanova | [@AlyonaCIA](https://github.com/AlyonaCIA) |
| Andrés Forero  | [@AndresF100](https://github.com/AndresF100) |
| Camilo Matson | [@camilomath](https://github.com/camilomath) |
| Santiago Calderón | [@SantiagoCalderonLopezUniandes](https://github.com/SantiagoCalderonLopezUniandes)|


## 📂 Estructura del Repositorio
```
└── 📁Repo desarrollo soluciones
    └── 📁.github
        └── 📁workflows
    └── 📁config
    └── 📁data
        └── 📁raw
    └── 📁deployment
    └── 📁Docs
    └── 📁experiments
    └── 📁notebooks
    └── 📁src
        └── 📁data
        └── 📁features
        └── 📁models
        └── 📁pipelines
        └── 📁utils
    └── 📁tests
    └── README.md
```

### Descripción de Carpetas

* data/ → Contiene los datos en diferentes estados del procesamiento. Se gestiona con DVC.

* notebooks/ → Notebooks de Jupyter para experimentación y análisis inicial.

* src/ → Código fuente organizado en módulos:

    * data/ → Scripts para cargar y preprocesar los datos.

    * features/ → Ingeniería de características.

    * models/ → Entrenamiento, evaluación y predicciones del modelo.

    * pipelines/ → Pipelines para procesar datos y entrenar modelos.

    * utils/ → Funciones auxiliares.

* Docs/ → Documentación del proyecto.

* experiments/ → Almacena resultados de experimentos y pruebas.

* deployment/ → Contiene la API de inferencia con FastAPI y archivos para el despliegue en producción.

* config/ → Parámetros de configuración del proyecto.

* tests/ → Pruebas unitarias y de integración para garantizar la calidad del código.

* .github/ → Workflows para CI/CD con GitHub Actions.



## 🚀 Requisitos

Si utilizas DVC, inicializa el proyecto con:
    
```bash
dvc init
dvc pull  # Para descargar los datos versionados
```