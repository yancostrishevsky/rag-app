"""Entrypoint for running the vector store service."""
import os

import uvicorn


def main() -> None:
    """Run the FastAPI app with uvicorn."""
    host = os.getenv('VECTOR_STORE_HOST', '0.0.0.0')
    port = int(os.getenv('VECTOR_STORE_PORT', '8000'))
    uvicorn.run('vector_store_service.app:app', host=host, port=port, reload=False)


if __name__ == '__main__':
    main()
