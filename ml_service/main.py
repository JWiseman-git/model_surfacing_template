import logging

from fastapi import FastAPI
import uvicorn

from ml_service.settings import get_settings
from ml_service import endpoints


LOGGER = logging.getLogger(__name__)


APP_NAME = "ml_service"
APP_PROJECT_DIR = "ml_service"
API_VERSION = "0.0.1"
SETTINGS = get_settings()


APP = FastAPI(
    title=APP_NAME,
    version=API_VERSION,
)
APP.include_router(endpoints.ROUTER)


def main():
    logging.basicConfig(level=logging.INFO)
    if SETTINGS.DEVELOPMENT_MODE:
        LOGGER.info("Starting in dev mode")
        uvicorn.run(
            "main:APP",
            host=SETTINGS.HOST,
            port=SETTINGS.PORT,
            log_level="info",
            workers=1,
            reload=True,
            reload_dirs=[APP_PROJECT_DIR],
        )
    else:
        LOGGER.info("Starting in non-dev mode")
        uvicorn.run(
            "main:APP",
            host=SETTINGS.HOST,
            port=SETTINGS.PORT,
            log_level="info",
            workers=1,
        )

if __name__ == "__main__":
    main()
