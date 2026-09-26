#!/usr/bin/env bash
# Copy the built binary and its runfiles to the Orin and run the matmul proof.
# The password file is never printed.

set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly HOST="${ORIN_HOST:-orin-1@orin-1.local}"
readonly PASSFILE="${ORIN_PASSWORD_FILE:-/mnt/workspace/orin-1.key}"
readonly DEST="${ORIN_DEST:-zml-on-arm64}"

cd "${ROOT}"

if [[ ! -e bazel-bin/zml_on_arm64 ]]; then
  echo "bazel-bin/zml_on_arm64 is missing. Run tools/build-arm64.sh first." >&2
  exit 1
fi
if [[ ! -d bazel-bin/zml_on_arm64.runfiles ]]; then
  echo "bazel-bin/zml_on_arm64.runfiles is missing. Run tools/build-arm64.sh first." >&2
  exit 1
fi

ASKPASS=""
cleanup() {
  if [[ -n "${ASKPASS}" ]]; then
    rm -f "${ASKPASS}"
  fi
}
trap cleanup EXIT

SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
if [[ -r "${PASSFILE}" ]]; then
  ASKPASS="$(mktemp)"
  cat > "${ASKPASS}" << EOF
#!/bin/sh
tr -d '\\r\\n' < '${PASSFILE}'
EOF
  chmod 700 "${ASKPASS}"
  export SSH_ASKPASS="${ASKPASS}" SSH_ASKPASS_REQUIRE=force
  export DISPLAY="${DISPLAY:-:0}"
  SSH+=(-o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no)
fi

RSYNC_RSH="${SSH[*]}"
export RSYNC_RSH

setsid -w "${SSH[@]}" "${HOST}" "mkdir -p ${DEST}"
setsid -w rsync -aL --info=stats2 \
  bazel-bin/zml_on_arm64 \
  bazel-bin/zml_on_arm64.runfiles \
  "${HOST}:${DEST}/"

setsid -w "${SSH[@]}" "${HOST}" "chmod +x ${DEST}/zml_on_arm64 && cd ${DEST} && ./zml_on_arm64 matmul"
