
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
    ├── 📁api
    ├── 📁config
    ├── 📁dashboard
    ├── 📁data
    ├── 📁dist
    ├── 📁docker
    ├── 📁Docs
    ├── 📁notebooks
    └── 📁src
```

### Descripción de Carpetas
* **api/**: Contiene el código de la API en FastAPI para exponer el modelo, incluyendo app.py y adaptadores.
* **config/**: Almacena archivos de configuración, credenciales (como dvc-key.json), y comandos para acceder a recursos externos (como instancias de EC2).
* **dashboard/**: Código del tablero interactivo para visualizar métricas y predicciones, organizado en `components/` y `utils/`.  
* **data/**: Contiene los datos del proyecto en diferentes fases (se obtiene por dvc):  
    - **raw/**: Datos originales sin procesar.  
    - **processed/**: Datos transformados para entrenar y validar el modelo.  
    - **visual/**: Datos específicos para visualización en el dashboard.
* **dist/**: Modelo empaquetado listo para instalación.
* **docker/**: Incluye Dockerfiles para construir imágenes de la API y el dashboard, facilitando el despliegue en contenedores.  
* **Docs/**: Documentación del proyecto, incluyendo análisis exploratorio y descripción del problema.  
* **notebooks/**: Notebooks de Jupyter para la exploración de datos, preprocesamiento y experimentos.  
* **src/**: Código fuente principal, dividido en submódulos:  
    - **data_preprocessing/**: Scripts para limpieza, transformación y creación del pipeline de preprocesamiento.  
    - **models/**: Implementación de distintos modelos (XGBoost, RandomForest, etc.), evaluaciones y búsqueda de hiperparámetros.  
    - **modelo_triage/**: Utilidades para cargar el modelo y transformar entradas antes de realizar predicciones.
 
  
## 🚀 Requisitos

Para descargar los datos, pide a uno de los creadores del repositorio que te comparta el **dvc-key.json**, agregalo a tu carpeta config, y ejecuta en dvc:
    
```bash
dvc remote modify gcsremote credentialpath ruta/a/dvc-key.json
```
