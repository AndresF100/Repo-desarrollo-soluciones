from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import random
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="Accident Prediction API", 
              description="API for predicting workplace accidents")

# Load the model
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/model.joblib")

# Try to load model at startup
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")
    model = None

# Define the input data model
class PredictionRequest(BaseModel):
    sector_economico: str  # Will map to parte_cuerpo in the dashboard
    genero: str  # Will map to municipio in the dashboard
    tipo_vinculacion: str  # Will map to tipo_jornada in the dashboard
    jornada_trabajo: str  # Will map to realizando_trabajo in the dashboard
    descripcion: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "sector_economico": "446",  # Example parte_cuerpo value
                "genero": "1",  # Example municipio value
                "tipo_vinculacion": "1",  # Example tipo_jornada value
                "jornada_trabajo": "SI",  # Example realizando_trabajo value
                "descripcion": "Trabajador cayó de una escalera mientras realizaba mantenimiento."
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    details: dict

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "warning", "message": "API is running but model is not loaded (placeholder mode active)"}
    return {"status": "ok", "message": "API is healthy and model is loaded"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict accident probability based on input features
    """
    # If no model is loaded, return a placeholder response
    if model is None:
        # Generate a random probability for the placeholder
        random_prob = round(random.uniform(0.3, 0.7), 2)
        
        return {
            "prediction": "Placeholder mientras pongo el modelo",
            "probability": random_prob,
            "details": {
                "input_features": request.dict(),
                "model_version": "placeholder",
                "note": "Este es un resultado provisional. El modelo real aún no está implementado."
            }
        }
    
    # If model is available, use it for prediction
    try:
        # Convert input data to dataframe format expected by the model
        input_data = pd.DataFrame({
            'sector_economico': [request.sector_economico],
            'genero': [request.genero],
            'tipo_vinculacion': [request.tipo_vinculacion],
            'jornada_trabajo': [request.jornada_trabajo],
            'descripcion': [request.descripcion if request.descripcion else ""]
        })
        
        # Make prediction
        prediction_label = "Positivo" if model.predict(input_data)[0] == 1 else "Negativo"
        probability = model.predict_proba(input_data)[0][1]  # Probability of positive class
        
        return {
            "prediction": prediction_label,
            "probability": float(probability),  # Convert numpy float to Python float
            "details": {
                "input_features": request.dict(),
                "model_version": getattr(model, "version", "unknown")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Add a reload endpoint for updating the model without restarting the API
@app.post("/reload-model")
async def reload_model():
    """Reload the model from disk"""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        return {"status": "warning", "message": f"Failed to load model: {str(e)}. Continuing in placeholder mode."}