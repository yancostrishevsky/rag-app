"""Pydantic models for the vector store service API."""
from typing import Dict
from typing import List
from typing import Tuple

import pydantic


class DocumentIn(pydantic.BaseModel):
    """Document payload for ingestion."""
    doc_id: str | None = None
    title: str
    content: str


class DocumentOut(pydantic.BaseModel):
    """Document payload for responses."""
    doc_id: str
    title: str
    content: str
    score: float | None = None


class RequestUpsertDocuments(pydantic.BaseModel):
    """Request to ingest or update documents."""
    documents: List[DocumentIn]


class ResponseUpsertDocuments(pydantic.BaseModel):
    """Response from ingestion."""
    upserted: int
    doc_ids: List[str]


class RequestCollectContextInfo(pydantic.BaseModel):
    """Request coming from the web app to collect context information."""
    user_message: str
    chat_history: List[Dict[str, str]]
    top_k: int = 5


class ResponseCollectContextInfo(pydantic.BaseModel):
    """Response sent back to the web app with collected context information."""
    context_docs: List[Tuple[str, str]]


class RequestStreamChatResponse(pydantic.BaseModel):
    """Request to stream a chat response."""
    user_message: str
    chat_history: List[Dict[str, str]]
    context_docs: List[Tuple[str, str]]


class RequestSearch(pydantic.BaseModel):
    """Request to retrieve similar documents."""
    query: str
    top_k: int = 5
    min_score: float = 0.0


class ResponseSearch(pydantic.BaseModel):
    """Response with ranked documents."""
    results: List[DocumentOut]
