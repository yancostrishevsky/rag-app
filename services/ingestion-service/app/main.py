import logging
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    documents: List[str]


class IngestResponse(BaseModel):
    ingested: int
    status: str


app = FastAPI(title="Ingestion Service")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest) -> IngestResponse:
    count = len(payload.documents)
    logger.info("Received %s documents for ingestion", count)
    return IngestResponse(ingested=count, status="ok")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
