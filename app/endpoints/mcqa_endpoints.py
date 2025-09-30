from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.model_ import MCQAModel

ROUTER = APIRouter(prefix="/mcqa", tags=["MCQA"])

mcqa_model = MCQAModel()

class MCQARequest(BaseModel):
    passage: str
    choices: list[str]

class MCQAResponse(BaseModel):
    prediction: str

@ROUTER.post("/predict", response_model=MCQAResponse)
def predict(request: MCQARequest):
    if len(request.choices) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 choices required")
    pred = mcqa_model.predict(request.passage, request.choices)
    return MCQAResponse(prediction=pred)
