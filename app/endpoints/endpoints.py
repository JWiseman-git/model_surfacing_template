from typing import Annotated
from collections import OrderedDict

import spacy

from fastapi import APIRouter, Depends
from pydantic import BaseModel


ROUTER = APIRouter()


class Request(BaseModel):
    text: str


class Response(BaseModel):
    chunks: list[str]


def get_chunks(text: str) -> list[str]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]


@ROUTER.get(
    "/chunkify",
    response_model=Response,
)
async def chunkify(params: Annotated[Request, Depends()]) -> Response:
    chunks = get_chunks(params.text)
    return Response(chunks=chunks)
