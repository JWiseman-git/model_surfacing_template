from pydantic import BaseModel
from typing import List, Dict

class MCQARequest(BaseModel):
    passage: str
    choices: list[str]

class MCQAResponse(BaseModel):
    prediction: Dict[str, float | str]

class MCQARequestBatch(BaseModel):
    passages: List[str]
    choices_list: List[List[str]]

class MCQAResponseBatch(BaseModel):
    predictions: List[Dict[str, float | str]]