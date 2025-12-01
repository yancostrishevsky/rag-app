# ------------------------------------------------------------------------
# This file contains the definition of a docker image used in course
# of the development process. The image built from this config contains
# the basic dependencies one needs in order to be able to setup the whole
# environment without installing anything locally on the system.
# ------------------------------------------------------------------------

FROM ubuntu:noble

SHELL ["/bin/bash", "-c"]

ARG USERNAME=devcontainer
ARG USER_UID=1000
ARG USER_GID=1000

COPY .devcontainer/scripts/base-setup.sh /tmp/base-setup.sh

RUN chmod +x /tmp/base-setup.sh && \
    USERNAME=${USERNAME} USER_UID=${USER_UID} USER_GID=${USER_GID} /tmp/base-setup.sh && \
    rm /tmp/base-setup.sh

WORKDIR /home/${USERNAME}/workspace
USER ${USERNAME}
