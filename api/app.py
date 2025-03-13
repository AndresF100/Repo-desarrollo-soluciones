from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import pandas as pd
import random
import traceback
import numpy as np  # Add NumPy import
import json
from typing import Optional, Dict, Any

# Define a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Initialize FastAPI app with custom JSON encoder
app = FastAPI(title="Accident Prediction API", 
              description="API for predicting workplace accidents")

# Add src directory to Python path if not already there
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added {src_dir} to Python path")

# Also add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added {parent_dir} to Python path")

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Import the data adapter
from adapter import create_model_input

# Model handling
model = None
model_info = {
    "name": "modelo_triage",
    "version": "1.0.0",
    "source_directory": "/app/src/modelo_triage",
    "load_error": None
}

# Try to import the model from src directory
try:
    print("Attempting to import model from modelo_triage.loader...")
    
    # Debug directory structure
    src_modelo_path = "/app/src/modelo_triage"
    if os.path.exists(src_modelo_path):
        print(f"Directory {src_modelo_path} exists. Contents:")
        for item in os.listdir(src_modelo_path):
            print(f" - {item}")
    else:
        print(f"Directory {src_modelo_path} does not exist")
        
    # Try checking for loader.py file
    loader_path = os.path.join(src_modelo_path, "loader.py")
    if os.path.exists(loader_path):
        print(f"Loader file exists at {loader_path}")
    else:
        print(f"Loader file not found at {loader_path}")
    
    # Import and load the model
    from modelo_triage.loader import get_model
    model = get_model()
    print(f"Model loaded successfully: {type(model)}")
except Exception as e:
    error_traceback = traceback.format_exc()
    print(f"Warning: Could not load model: {str(e)}")
    print(f"Traceback: {error_traceback}")
    model_info["load_error"] = str(e)
    model = None

# Define the input data model
class PredictionRequest(BaseModel):
    parte_cuerpo: str
    municipio: str
    jornada_trabajo: str
    realizando_trabajo: str
    descripcion: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "parte_cuerpo": "446",
                "municipio": "1",
                "jornada_trabajo": "1",
                "realizando_trabajo": "s",
                "descripcion": "Trabajador cayó de una escalera mientras realizaba mantenimiento."
            }
        }

class PredictionResponse(BaseModel):
    prediction: Any
    details: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Accident Prediction API",
        "status": "online",
        "model_loaded": model is not None,
        "model_info": model_info,
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/predict", "method": "POST", "description": "Make predictions"},
            {"path": "/reload-model", "method": "POST", "description": "Reload model"},
            {"path": "/debug", "method": "GET", "description": "Debug information"}
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "warning", "message": "API is running but model is not loaded (placeholder mode active)", "error": model_info.get("load_error")}
    return {"status": "ok", "message": "API is healthy and model is loaded", "model_info": model_info}

@app.get("/debug")
async def debug_info():
    """Provide debugging information"""
    debug_data = {
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "sys_path": sys.path,
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model else None,
        "model_info": model_info,
        "environment": {k: v for k, v in os.environ.items() if not k.startswith("_") and k.lower() not in ("key", "secret", "password", "token")}
    }
    
    # Check for src/modelo_triage directory
    src_model_path = "/app/src/modelo_triage"
    debug_data["src_modelo_exists"] = os.path.exists(src_model_path)
    
    if debug_data["src_modelo_exists"]:
        debug_data["src_modelo_contents"] = os.listdir(src_model_path)
        
    # Try importing the modelo_triage module
    try:
        import modelo_triage
        debug_data["modelo_triage_imported"] = True
        debug_data["modelo_triage_path"] = modelo_triage.__file__
    except ImportError as e:
        debug_data["modelo_triage_imported"] = False
        debug_data["import_error"] = str(e)
        
    return debug_data

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict accident probability based on input features
    """
    # If no model is loaded, return a placeholder response
    if model is None:
        return {
            "prediction": "Placeholder mientras pongo el modelo",
            "details": {
                "input_features": request.dict(),
                "model_version": "placeholder",
                "note": "Este es un resultado provisional. El modelo real aún no está implementado.",
                "error": model_info.get("load_error", "Unknown error")
            }
        }
    
    # If model is available, use it for prediction
    try:
        # Convert API request to model input format using the adapter
        api_input = request.dict()
        model_input = create_model_input(api_input)
        
        # Make prediction with properly formatted input
        prediction_result = model.predict(model_input)[0]
        
        # Convert NumPy types to Python native types
        if isinstance(prediction_result, np.integer):
            prediction_result = int(prediction_result)
        elif isinstance(prediction_result, np.floating):
            prediction_result = float(prediction_result)
        
        # Convert any NumPy values in adapted_fields to Python native types
        adapted_fields = {}
        for k, v in model_input.iloc[0].items():
            if k in [
                'id_parte_cuerpo_igatepmafurat',
                'id_municipio_at_igatepmafurat',
                'ind_tipo_jornada_at_igatepmafurat',
                'ind_realizando_trabajo_hab_at_igatepmafurat',
                'descripcion_at_igatepmafurat'
            ]:
                if isinstance(v, (np.integer, np.floating)):
                    adapted_fields[k] = int(v) if isinstance(v, np.integer) else float(v)
                else:
                    adapted_fields[k] = v
        
        return {
            "prediction": prediction_result,
            "details": {
                "input_features": request.dict(),
                "model_info": model_info,
                "adapted_fields": adapted_fields
            }
        }
    except Exception as e:
        error_traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}\n{error_traceback}")

@app.post("/reload-model")
async def reload_model():
    """Reload the model from the src directory"""
    global model, model_info
    try:
        # Force reload the module
        import importlib
        
        # If module is in sys.modules, reload it
        if 'modelo_triage' in sys.modules:
            print("Reloading modelo_triage module")
            importlib.reload(sys.modules['modelo_triage'])
            if 'modelo_triage.loader' in sys.modules:
                importlib.reload(sys.modules['modelo_triage.loader'])
        
        # Load the model
        from modelo_triage.loader import get_model
        model = get_model()
        model_info["load_error"] = None
        return {"status": "success", "message": "Model reloaded successfully", "model_type": str(type(model))}
    except Exception as e:
        error_traceback = traceback.format_exc()
        model = None
        model_info["load_error"] = str(e)
        return {"status": "error", "message": f"Failed to reload model: {str(e)}", "traceback": error_traceback}