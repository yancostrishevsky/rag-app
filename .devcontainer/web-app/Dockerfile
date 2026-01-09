FROM python:3.12-slim

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      wget \
      git \
      openssh-client \
      sudo \
      just \
      libmagic1

RUN groupadd -g 1000 appuser && useradd appuser -u 1000 -g 1000 -m -s /bin/bash
RUN chown -R appuser:appuser /home/appuser/

COPY .devcontainer/scripts/base-setup.sh /tmp/base-setup.sh
RUN chmod +x /tmp/base-setup.sh \
    && USERNAME="appuser" /tmp/base-setup.sh

USER appuser
WORKDIR /home/appuser/workspace

RUN wget -qO- https://astral.sh/uv/install.sh | sh