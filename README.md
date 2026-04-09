# TP7 - MLOps: CI/CD, S3 y Reentrenamiento

## Descripción
API de Machine Learning para predicción de precios de viviendas con:
- CI/CD con GitHub Actions
- Almacenamiento en AWS S3
- Reentrenamiento automático diario

## Tecnologías
- FastAPI
- AWS S3
- GitHub Actions
- Scikit-learn

## Endpoints
- `GET /health` - Health check
- `POST /predict` - Realizar predicción
- `POST /train` - Reentrenar modelo

## Configuración
1. Clonar repositorio
2. Crear archivo `.env` con credenciales AWS
3. `pip install -r requirements.txt`
4. `python scripts/train.py`
5. `uvicorn app.main:app --reload`

## CI/CD
El pipeline se ejecuta en cada push y entrena/subel modelo a S3 automáticamente.


Ivan Cespedes
