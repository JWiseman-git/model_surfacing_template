from fastapi import APIRouter, HTTPException
from app.schemas import MCQARequest, MCQAResponse, ChoiceScore, MCQAResponseScores
from app.model import MCQAModel

ROUTER = APIRouter(prefix="/mcqa", tags=["MCQA"])

mcqa_model = MCQAModel()

@ROUTER.post("/predict", response_model=MCQAResponse)
def predict(request: MCQARequest):
    if len(request.choices) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 choices required")
    pred = mcqa_model.predict_blank(request.passage, request.choices)
    return MCQAResponse(prediction=pred)

@ROUTER.post("/predict_with_scores", response_model=MCQAResponseScores)
def predict_with_scores(request: MCQARequest):
    results = mcqa_model.predict_blank(request.passage, request.choices, return_scores=True)
    return {"top_choices": results}

