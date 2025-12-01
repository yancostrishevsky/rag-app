from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

CORE_SERVICE_URL = "http://core-service:8000/query"


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str


app = FastAPI(title="API Gateway")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(CORE_SERVICE_URL, json=payload.dict())
            response.raise_for_status()
    except httpx.HTTPError as exc:  # pragma: no cover - placeholder error handling
        raise HTTPException(status_code=502, detail=f"core-service error: {exc}") from exc

    data = response.json()
    return AskResponse(answer=data.get("answer", ""))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
