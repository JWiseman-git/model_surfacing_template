import logging
import mlflow
import mlflow.pytorch

from app.models.mcqa_model import MCQAModel, MCQAConfig
from app.settings import settings

logger = logging.getLogger(__name__)

def load_mcqa_model():
    """
    Load the MCQA model locally (optionally log it to a local MLflow run for demonstration)

    Steps:
    1. Load pre-trained model from the local directory.
    2. Start a local MLflow run and log the model as an artifact.
    3. Return the loaded MCQA model for further use.
    """
    logger.info(f"Loading MCQA model from {settings.model_directory}")
    config = MCQAConfig(model_directory=settings.model_directory)
    model = MCQAModel(config)

    with mlflow.start_run(run_name="MCQA_Dev_Save") as run:
        logger.info("Logging model to MLflow (local run only)")
        mlflow.pytorch.log_model(
            pytorch_model=model.model,
            name="MCQAModel_dev"
        )

        logger.info("Model logged as artifact 'MCQAModel_local' in local MLflow run")

    return model

    # Note: This code may be used later for an approach making use of a server side model registry
    # try:
    #     model_uri = f"MCQAModel_dev"
    #     result = mlflow.register_model(
    #         model_uri=model_uri,
    #         name="MCQAModel_dev"
    #     )
    #     logger.info(f"Model registered in MLflow registry: {result.name}")
    # except Exception as e:
    #     logger.warning(f"Model registry not configured or registration failed: {e}")

    return model

if __name__ == "__main__":
    mcqa_model = load_mcqa_model()
    logger.info("MCQA model loaded successfully")