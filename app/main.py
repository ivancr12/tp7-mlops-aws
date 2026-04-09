from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import HousingPriceModel
from app.s3_utils import S3Utils
from app.config import Config

app = FastAPI(title="MLOps TP7", description="API con S3 integration")

# Variables globales
model = HousingPriceModel()
s3 = S3Utils()

class PredictionRequest(BaseModel):
    features: list[float]

@app.on_event("startup")
async def startup_event():
    """Carga el modelo desde S3 al iniciar"""
    try:
        if s3.download_file(Config.MODEL_PATH, 'models/model_tmp.pkl'):
            model.load('models/model_tmp.pkl')
            print("✅ Modelo cargado desde S3")
        else:
            print("⚠️ No se pudo cargar modelo de S3")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

@app.get("/")
def root():
    return {
        "message": "MLOps TP7 - API con S3",
        "status": "running",
        "endpoints": ["/health", "/predict", "/train"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Intentar cargar modelo si no está disponible
        if model.model is None:
            if s3.download_file(Config.MODEL_PATH, 'models/model_tmp.pkl'):
                model.load('models/model_tmp.pkl')
        
        prediction = model.predict(request.features)
        return {
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train():
    """Endpoint para entrenar y subir a S3"""
    import subprocess
    result = subprocess.run(['python', 'scripts/train.py'], capture_output=True, text=True)
    return {
        "message": "Entrenamiento completado",
        "output": result.stdout,
        "timestamp": datetime.now().isoformat()
    }
