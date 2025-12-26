FROM python:3.12-slim

WORKDIR /app

COPY services/vector-store-service/pyproject.toml /app/pyproject.toml
COPY services/vector-store-service/src /app/src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir fastapi==0.115.12 uvicorn==0.34.3

ENV PYTHONPATH=/app/src
ENV VECTOR_STORE_HOST=0.0.0.0
ENV VECTOR_STORE_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "vector_store_service"]
