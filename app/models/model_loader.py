import mlflow
from app.models.mcqa_model import MCQAModel
from app.settings import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

# todo add logging here

def load_mcqa_model():
    """
    One-time script to load and save the MCQA model to a local MLflow tracking server.

    This script performs the following steps:
    1. Loads a pre-trained Multiple-Choice Question Answering (MCQA) model from the local directory
       specified in `settings.model_directory`.
    2. Starts an MLflow run named "MCQA_Dev_Save".
    3. Logs the PyTorch model to MLflow under the name "MCQAModel_dev".
    4. Ends the MLflow run automatically when complete.
    5. Returns the loaded MCQA model for optional further use.

    Usage:
        python -m app.models.model_loader

    Notes:
    - This script is intended for **development or testing**. It creates a one-time saved version of
      the model in the MLflow tracking directory (`settings.mlflow_tracking_uri`).
    """

    model = MCQAModel(model_directory=settings.model_directory)
    with mlflow.start_run(run_name="MCQA_Dev_Save") as run:
        mlflow.pytorch.log_model(
            pytorch_model=model.model,
            name="MCQAModel_dev"
        )
        run_id = run.info.run_id

    mlflow.register_model(
        f"runs:/{run_id}/MCQAModel_dev",
        "MCQAModel_dev"
    )
    return model

mcqa_model = load_mcqa_model()

if __name__ == "__main__":
    load_mcqa_model()