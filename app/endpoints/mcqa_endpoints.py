import mlflow
from fastapi import APIRouter, HTTPException
from app.schemas.schemas import MCQARequest, MCQAResponse, MCQARequestBatch, MCQAResponseBatch
from app.models.mcqa_model import MCQAModel, MCQAConfig
from app.settings import settings

ROUTER = APIRouter(prefix="/mcqa", tags=["MCQA"])

# This would be enabled when handling loading from a model registry
# artifact_path = "mlruns/0/<run_id>/artifacts/MCQAModel_local"
# mcqa_model = mlflow.pytorch.load_model(artifact_path)

config = MCQAConfig(model_directory=settings.model_directory)
mcqa_model = MCQAModel(config)

@ROUTER.post("/predict", response_model=MCQAResponse)
def predict(request: MCQARequest):
    """
    This endpoint expects a passage and exactly four answer choices. It uses the
    MCQA model to select the most likely correct choice.

    Parameters:
    - request (MCQARequest): Input data containing:
        - passage (str): The text passage containing the blank or context.
        - choices (List[str]): A list of exactly 4 possible answer choices.

    Returns:
    - MCQAResponse: The model's predicted answer.

    Raises:
    - HTTPException: If the number of choices is not exactly 4.
    """
    if len(request.choices) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 choices required")
    pred = mcqa_model.predict_blank(request.passage, request.choices)
    return MCQAResponse(prediction=pred)

@ROUTER.post("/predict_chunk", response_model=MCQAResponseBatch)
def predict_chunk(request: MCQARequestBatch):
    """
    This endpoint expects multiple passages, each with exactly four answer choices.
    It uses the MCQA model to select the most likely correct choice for each chunk.

    Parameters:
    - request (MCQARequestBatch): Input data containing:
        - passages (List[str]): List of text chunks containing [BLANK] placeholders.
        - choices_list (List[List[str]]): List of 4-choice lists, one per passage.

    Returns:
    - MCQAResponseBatch: The model's predicted answers for each chunk.

    Raises:
    - HTTPException: If lengths mismatch or choices per chunk are not exactly 4.
    """
    if len(request.passages) != len(request.choices_list):
        raise HTTPException(
            status_code=400,
            detail="Number of passages does not match number of choices lists"
        )

    preds = mcqa_model.predict_batch(request.passages, request.choices_list)
    return MCQAResponseBatch(predictions=preds)
