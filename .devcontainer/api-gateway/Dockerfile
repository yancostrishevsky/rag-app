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
      python3-pip && \
    rm -rf /var/lib/apt/lists/*

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace
ENV PATH="/home/devcontainer/.local/bin:${PATH}"

