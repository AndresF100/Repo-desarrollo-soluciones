
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
    └── 📁config
    └── 📁dashboard
    └── 📁api
    └── 📁docker
    └── 📁data
    └── 📁Docs
    └── 📁notebooks
    └── 📁src
```

### Descripción de Carpetas

* **config/**: Contiene configuraciones, llaves de acceso para sesiones de VM y scripts auxiliares.
* **.github/**: Almacena configuraciones para automatización con GitHub Actions (CI/CD).
* **notebooks/**: Incluye notebooks de Jupyter para exploración, preprocesamiento y análisis.
* **Docs/**: Documentación del proyecto con informes y análisis del problema.
* **data/**: Almacena los datos en distintas etapas: crudos (raw), procesados (processed), y para visualización (visual).
* **src/**: Código fuente del proyecto, dividido en submódulos para preprocesamiento (data_preprocessing/) y modelos (models/).
* **dashboard/**: Código fuente del dashboard, incluyendo componentes/utilidades.
* **api/**: Código fuente de la API que permite consumir el modelo a través del tablero.
* **docker/**: Código fuente del despliegue del tablero y la API en docker, incluyendo DOCKERFILE.

## 🚀 Requisitos

Si utilizas DVC, inicializa el proyecto con:
    
```bash
dvc init
dvc pull
```
