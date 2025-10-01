import uvicorn
import logging
import os

from fastapi import FastAPI
from utils.logger import setup_logging
from app.endpoints.mcqa_endpoints import ROUTER as MCQA_ROUTER

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCQA API")
app.include_router(MCQA_ROUTER)

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8080))

    logger.info("Application started on %s:%s", host, port)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=os.getenv("ENV", "dev") == "dev"
    )