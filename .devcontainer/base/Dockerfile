FROM ubuntu:noble

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      wget \
      git \
      sudo \
      just \
      openssh-client \
      jq \
      python3 \
      python3-pip \
      docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages docker-compose

ARG USERNAME=base

RUN if ! id "${USERNAME}" >/dev/null 2>&1; then \
      groupadd "${USERNAME}" || true; \
      useradd -m -s /bin/bash -g "${USERNAME}" "${USERNAME}"; \
    fi

COPY .devcontainer/scripts/base-setup.sh /tmp/base-setup.sh
RUN chmod +x /tmp/base-setup.sh \
    && USERNAME="${USERNAME}" /tmp/base-setup.sh \
    && rm /tmp/base-setup.sh

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
