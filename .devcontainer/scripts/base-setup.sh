set -euo pipefail

USERNAME="${USERNAME:-base}"

if ! id "${USERNAME}" &>/dev/null; then
  echo "User '${USERNAME}' does not exist. Create it in the Dockerfile before running this script." >&2
  exit 1
fi

usermod -aG sudo "${USERNAME}"
echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"
chmod 440 "/etc/sudoers.d/${USERNAME}"

git config --system core.sshCommand /usr/bin/ssh
