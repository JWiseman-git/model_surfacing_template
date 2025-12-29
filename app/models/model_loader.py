import logging

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


    return model


if __name__ == "__main__":
    mcqa_model = load_mcqa_model()
    logger.info("MCQA model loaded successfully")