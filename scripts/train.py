import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import HousingPriceModel
from app.s3_utils import S3Utils
from app.config import Config

def generate_data():
    np.random.seed(42)
    size = np.random.normal(150, 50, 500)
    bedrooms = np.random.randint(1, 5, 500)
    age = np.random.uniform(0, 50, 500)
    
    price = 3000*size + 20000*bedrooms - 1000*age + np.random.normal(0, 50000, 500)
    
    X = pd.DataFrame({'size': size, 'bedrooms': bedrooms, 'age': age})
    y = price
    return X, y

def main():
    print("🚀 Iniciando entrenamiento...")
    
    # Inicializar S3
    s3 = S3Utils()
    
    # Generar o cargar datos
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar
    model = HousingPriceModel()
    metrics = model.train(X_train, y_train)
    
    # Guardar localmente
    os.makedirs('models', exist_ok=True)
    model.save('models/model.pkl')
    print(f"📊 Score: {metrics['score']}")
    
    # Subir a S3
    s3.upload_file('models/model.pkl', Config.MODEL_PATH)
    
    print("✅ Entrenamiento completado!")

if __name__ == "__main__":
    main()
