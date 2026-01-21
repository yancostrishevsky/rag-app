FROM ubuntu:noble

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      wget \
      git \
      openssh-client \
      sudo \
      just

COPY .devcontainer/scripts/base-setup.sh /tmp/base-setup.sh
RUN chmod +x /tmp/base-setup.sh \
    && USERNAME="ubuntu" /tmp/base-setup.sh

USER ubuntu
WORKDIR /home/ubuntu/workspace
