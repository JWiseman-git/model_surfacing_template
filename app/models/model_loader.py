import logging
import mlflow

from app.models.mcqa_model import MCQAModel, MCQAConfig
from app.settings import settings

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def load_mcqa_model():
    """
    One-time script to load and save the MCQA model to a local MLflow tracking server.

    This script performs the following steps:
    1. Loads a pre-trained  model from the local directory.
    2. Starts an MLflow run named "MCQA_Dev_Save".
    3. Logs the model to MLflow.
    4. Ends the MLflow run automatically when complete.
    5. Returns the loaded MCQA model for optional further use.

    Usage:
        python -m app.models.model_loader

    Notes:
    - This script is intended for **development or testing**. It creates a one-time saved version of
      the model in the MLflow tracking directory (`settings.mlflow_tracking_uri`).
    """
    logger.info(f"Loading MCQA model from {settings.model_directory}")
    config = MCQAConfig(model_directory=settings.model_directory)
    model = MCQAModel(config)

    with mlflow.start_run(run_name="MCQA_Dev_Save") as run:
        logger.info("Logging model to MLflow under artifact path 'MCQAModel_dev'")
        mlflow.pytorch.log_model(
            pytorch_model=model.model,
            name="MCQAModel_dev"
        )
        run_id = run.info.run_id

        logger.info(f"Model logged with run_id {run_id}")

    try:
        model_uri = f"runs:/{run_id}/MCQAModel_dev"
        result = mlflow.register_model(
            model_uri=model_uri,
            name="MCQAModel_dev"
        )
        logger.info(f"Model registered in MLflow registry: {result.name}")
    except Exception as e:
        logger.warning(f"Model registry not configured or registration failed: {e}")

    return model

if __name__ == "__main__":
    load_mcqa_model()