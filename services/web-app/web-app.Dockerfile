FROM python:3.12-slim

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget libmagic1

RUN wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

COPY cfg /app/cfg
COPY src /app/src
COPY pyproject.toml main.py /app/
WORKDIR /app

EXPOSE ${WEB_APP_PORT}

RUN uv python install 3.12 && uv venv && uv pip install .
CMD [ "uv", "run", "python", "main.py" ]
