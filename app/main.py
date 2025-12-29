import uvicorn
import logging
import mlflow

from fastapi import FastAPI
from app.utils.logger import setup_logging
from app.endpoints.mcqa_endpoints import ROUTER as MCQA_ROUTER
from app.settings import settings

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCQA API")
app.include_router(MCQA_ROUTER)

if __name__ == "__main__":
    logger.info(f"Application started on {settings.host}, {settings.port}")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.env == "dev"
    )