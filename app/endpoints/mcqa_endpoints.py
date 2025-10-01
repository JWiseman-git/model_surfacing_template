from fastapi import APIRouter, HTTPException
from app.schemas import MCQARequest, MCQAResponse, ChoiceScore, MCQAResponseScores
from app.model import MCQAModel

ROUTER = APIRouter(prefix="/mcqa", tags=["MCQA"])

mcqa_model = MCQAModel()

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

