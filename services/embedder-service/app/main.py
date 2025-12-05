from typing import List
from fastapi import FastAPI
from pydantic import BaseModel


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


app = FastAPI(title="Embedder Service")


@app.post("/embed", response_model=EmbedResponse)
async def embed(payload: EmbedRequest) -> EmbedResponse:
    embeddings = [[float(len(text))] for text in payload.texts]
    return EmbedResponse(embeddings=embeddings)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
