FROM python:3.12-slim

WORKDIR /app

COPY services/vector-store-service/pyproject.toml /app/pyproject.toml
COPY services/vector-store-service/src /app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi==0.115.12 \
        faiss-cpu==1.8.0 \
        numpy==1.26.4 \
        sentence-transformers==2.7.0 \
        uvicorn==0.34.3

ENV PYTHONPATH=/app/src
ENV VECTOR_STORE_HOST=0.0.0.0
ENV VECTOR_STORE_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "vector_store_service"]
