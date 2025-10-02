from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8080
    env: str = "dev"
    mlflow_tracking_uri: str = "file:./mlruns_dev"
    model_directory: str = "./models/mcqa"

settings = Settings()
