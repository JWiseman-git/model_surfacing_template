import uvicorn
import logging
import mlflow

from fastapi import FastAPI
from utils.logger import setup_logging
from app.endpoints.mcqa_endpoints import ROUTER as MCQA_ROUTER
from app.settings import settings

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCQA API")
app.include_router(MCQA_ROUTER)

if __name__ == "__main__":
    logger.info("Application started on %s:%s", settings.host, settings.port)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with mlflow.start_run(run_name=f"MCQA_API_Run_{settings.env}") as run:
        logger.info(f"MLflow run started: {run.info.run_id}")
        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.env == "dev"
        )