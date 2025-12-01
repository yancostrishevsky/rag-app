#!/usr/bin/env bash
set -euo pipefail

USERNAME="${USERNAME:-devcontainer}"
USER_UID="${USER_UID:-1000}"
USER_GID="${USER_GID:-1000}"

export DEBIAN_FRONTEND=noninteractive

# Basic sanity: remove default ubuntu user if present
if id ubuntu &>/dev/null; then
  userdel -r ubuntu || true
  groupdel ubuntu || true
fi

# Create group & user
if ! getent group "${USER_GID}" >/dev/null 2>&1; then
  groupadd -g "${USER_GID}" "${USERNAME}"
fi

if ! id "${USERNAME}" >/dev/null 2>&1; then
  useradd -m -s /bin/bash -u "${USER_UID}" -g "${USER_GID}" "${USERNAME}"
fi

# sudo, but no password prompts inside the container
apt-get update
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
  python3-pip

rm -rf /var/lib/apt/lists/*

usermod -aG sudo "${USERNAME}"
echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"
chmod 440 "/etc/sudoers.d/${USERNAME}"

git config --system core.sshCommand /usr/bin/ssh

chown -R "${USERNAME}:${USERNAME}" "/home/${USERNAME}"
mkdir -p "/home/${USERNAME}/workspace"
chown -R "${USERNAME}:${USERNAME}" "/home/${USERNAME}/workspace"
