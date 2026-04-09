import os

class Config:
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET = os.getenv('S3_BUCKET', 'tp7-mlops-bucket')
    MODEL_PATH = 'models/model.pkl'
    DATA_PATH = 'data/housing_data.csv'
