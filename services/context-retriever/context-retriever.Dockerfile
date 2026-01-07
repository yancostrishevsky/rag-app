FROM python:3.12-slim

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget

RUN wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

COPY . /app
WORKDIR /app

EXPOSE ${API_PORT}

RUN uv python install 3.12 && uv venv && uv pip install .
CMD [ "uv", "run", "python", "main.py" ]