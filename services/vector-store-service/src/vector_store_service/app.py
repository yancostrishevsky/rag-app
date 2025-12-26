"""FastAPI app exposing document ingestion and retrieval."""
import asyncio
import json
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from vector_store_service.schemas import DocumentOut
from vector_store_service.schemas import RequestCollectContextInfo
from vector_store_service.schemas import RequestStreamChatResponse
from vector_store_service.schemas import RequestSearch
from vector_store_service.schemas import RequestUpsertDocuments
from vector_store_service.schemas import ResponseCollectContextInfo
from vector_store_service.schemas import ResponseSearch
from vector_store_service.schemas import ResponseUpsertDocuments
from vector_store_service.settings import load_settings
from vector_store_service.vector_store import Document
from vector_store_service.vector_store import VectorStore


settings = load_settings()
vector_store = VectorStore(
    db_path=str(settings.db_path),
    embedding_dim=settings.embedding_dim
)

app = FastAPI(title='vector-store-service')


@app.get('/ping')
def ping() -> dict[str, str]:
    """Health check endpoint."""
    return {'message': 'Service is running'}


@app.post('/documents', response_model=ResponseUpsertDocuments)
def upsert_documents(request: RequestUpsertDocuments) -> ResponseUpsertDocuments:
    """Stores documents in the vector store."""
    documents = []
    for document in request.documents:
        doc_id = document.doc_id or str(uuid.uuid4())
        documents.append(Document(
            doc_id=doc_id,
            title=document.title,
            content=document.content
        ))

    doc_ids = vector_store.upsert_documents(documents)
    return ResponseUpsertDocuments(upserted=len(doc_ids), doc_ids=doc_ids)


@app.post('/collect_context_info', response_model=ResponseCollectContextInfo)
def collect_context_info(
    request: RequestCollectContextInfo
) -> ResponseCollectContextInfo:
    """Returns top documents for the user message."""
    results = vector_store.search(
        query=request.user_message,
        top_k=request.top_k,
        min_score=settings.min_score
    )
    context_docs = [(doc.title, doc.content) for doc, _ in results]
    return ResponseCollectContextInfo(context_docs=context_docs)


@app.post('/stream_chat_response')
async def stream_chat_response(
    request: RequestStreamChatResponse
) -> StreamingResponse:
    """Streams a placeholder response listing the retrieved documents."""
    titles = [title for title, _ in request.context_docs]
    response = (
        f"I found {len(titles)} documents for your query. "
        f"Top matches: {', '.join(titles) if titles else 'none'}."
    )

    async def event_generator():
        parts = response.split()
        for index, part in enumerate(parts):
            token = part if index == 0 else f" {part}"
            yield json.dumps({'content': token}).encode('utf-8')
            await asyncio.sleep(0.02)

    return StreamingResponse(event_generator(), media_type='application/json')


@app.post('/search', response_model=ResponseSearch)
def search(request: RequestSearch) -> ResponseSearch:
    """Returns ranked documents with scores."""
    results = vector_store.search(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score
    )
    return ResponseSearch(
        results=[
            DocumentOut(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                score=score
            )
            for doc, score in results
        ]
    )
