from pathlib import Path
from setuptools import find_packages, setup

# Metadatos del paquete
NAME = 'model-triage'
DESCRIPTION = "Modelo de clasificación de accidentes laborales."
URL = ""
EMAIL = "af.foreroo1@uniandes.edu.co"
AUTHOR = "andresforero"
REQUIRES_PYTHON = ">=3.10.0"

about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "src"

# Leer la versión desde el archivo VERSION
with open(PACKAGE_DIR / "VERSION") as f:
    about["__version__"] = f.read().strip()

# Lista de dependencias
def list_reqs(fname="requirements.txt"):
    with open(PACKAGE_DIR / fname) as fd:
        return fd.read().splitlines()

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where="src", include=["*"]),
    package_dir={"": "src"},
    install_requires=list_reqs(),
    include_package_data=True,  # 🔥 Incluir archivos no-Python
    package_data={
        "": [
            "VERSION",                  # Asegura la inclusión del archivo VERSION
            "requirements.txt",         # Asegura la inclusión de las dependencias
            "**/*.pkl",                 # 🔥 Incluye todos los archivos .pkl
            
        ],
    },
    license="BSD-3",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
