from fastapi import FastAPI
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


def run_query(query: str) -> str:
    return f"dummy answer for: {query}"


app = FastAPI(title="Core Service")


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    answer = run_query(payload.query)
    return QueryResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
