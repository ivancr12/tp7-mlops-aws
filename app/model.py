import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

class HousingPriceModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def train(self, X_train, y_train):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = LinearRegression()
        self.model.fit(X_scaled, y_train)
        
        return {'score': float(self.model.score(X_scaled, y_train))}
    
    def predict(self, features):
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        return float(self.model.predict(features_scaled)[0])
    
    def save(self, path):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
