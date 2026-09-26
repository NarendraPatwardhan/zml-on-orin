#!/usr/bin/env bash
# Schedule the aarch64 CUDA build on a BuildBuddy remote runner.
# This machine does not run Bazel.

set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to build: commit first so the remote run matches git." >&2
  exit 1
fi

exec bb remote \
  --os=linux \
  --arch=arm64 \
  --timeout=3h \
  --runner_exec_properties=recycle-runner=false \
  --runner_exec_properties=EstimatedFreeDiskBytes=80GB \
  build \
  --config=buildbuddy \
  --config=release \
  //:zml_on_arm64
