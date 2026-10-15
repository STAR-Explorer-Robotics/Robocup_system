#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  exec "$@"
fi

HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"
CONTAINER_USER="qrc"
CONTAINER_GROUP="qrc"

if ! getent group "${CONTAINER_GROUP}" >/dev/null; then
  groupadd -g "${HOST_GID}" "${CONTAINER_GROUP}"
else
  groupmod -g "${HOST_GID}" "${CONTAINER_GROUP}"
fi

if ! id -u "${CONTAINER_USER}" >/dev/null 2>&1; then
  useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${CONTAINER_USER}"
else
  usermod -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${CONTAINER_USER}"
fi

home_dir="/home/${CONTAINER_USER}"
mkdir -p "${home_dir}"
chown "${HOST_UID}:${HOST_GID}" "${home_dir}"

for path in \
  /workspace/qrc_ros2_perception/build \
  /workspace/qrc_ros2_perception/install \
  /workspace/qrc_ros2_perception/log \
  /workspace/qrc_ros2_perception/outputs \
  /workspace/qrc_ros2_perception/recordings; do
  mkdir -p "${path}"
  chown -R "${HOST_UID}:${HOST_GID}" "${path}" 2>/dev/null || true
done

export HOME="${home_dir}"
export USER="${CONTAINER_USER}"
export LOGNAME="${CONTAINER_USER}"

exec gosu "${HOST_UID}:${HOST_GID}" "$@"
