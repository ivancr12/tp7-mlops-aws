import boto3
import joblib
import pandas as pd
import io
import os
from botocore.exceptions import ClientError
from .config import Config

class S3Utils:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=Config.AWS_ACCESS_KEY,
            aws_secret_access_key=Config.AWS_SECRET_KEY,
            region_name=Config.AWS_REGION
        )
        self.bucket = Config.S3_BUCKET
    
    def upload_file(self, local_path, s3_path):
        try:
            self.s3_client.upload_file(local_path, self.bucket, s3_path)
            print(f"✅ Subido: {local_path} -> s3://{self.bucket}/{s3_path}")
            return True
        except ClientError as e:
            print(f"❌ Error: {e}")
            return False
    
    def download_file(self, s3_path, local_path):
        try:
            self.s3_client.download_file(self.bucket, s3_path, local_path)
            print(f"✅ Descargado: s3://{self.bucket}/{s3_path} -> {local_path}")
            return True
        except ClientError as e:
            print(f"❌ Error: {e}")
            return False
    
    def file_exists(self, s3_path):
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_path)
            return True
        except ClientError:
            return False
