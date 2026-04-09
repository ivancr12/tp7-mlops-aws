from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "MLOps API funcionando", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
def predict(request: PredictionRequest):
    return {
        "prediction": sum(request.features),
        "timestamp": datetime.now().isoformat()
    }
