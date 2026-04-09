import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import HousingPriceModel

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
    print("Entrenando modelo...")
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = HousingPriceModel()
    metrics = model.train(X_train, y_train)
    
    os.makedirs('model', exist_ok=True)
    model.save('model/model.pkl')
    
    print(f"Modelo guardado. Score: {metrics['score']}")
    return metrics

if __name__ == "__main__":
    main()
