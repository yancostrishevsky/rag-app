FROM ubuntu:noble

SHELL ["/bin/bash", "-c"]

ARG USERNAME=devcontainer
ARG USER_UID=1000
ARG USER_GID=1000

COPY .devcontainer/scripts/base-setup.sh /tmp/base-setup.sh
RUN chmod +x /tmp/base-setup.sh && \
    USERNAME=${USERNAME} USER_UID=${USER_UID} USER_GID=${USER_GID} /tmp/base-setup.sh && \
    rm /tmp/base-setup.sh

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/${USERNAME}/workspace/services/api-gateway

COPY services/api-gateway/requirements.txt ./requirements.txt

RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:${PATH}"

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace

ENV PATH="/home/devcontainer/.local/bin:${PATH}"
