from pydantic import BaseModel

from typing import List

class MCQARequest(BaseModel):
    passage: str
    choices: list[str]

class MCQAResponse(BaseModel):
    prediction: str

class ChoiceScore(BaseModel):
    choice: str
    score: float

class MCQAResponseScores(BaseModel):
    top_choices: List[ChoiceScore]