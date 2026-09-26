#!/usr/bin/env bash
# The aarch64 build is a BuildBuddy workflow, triggered by pushing master.
# Do not invoke Bazel or bb on this machine.

set -euo pipefail

echo "The arm64 build runs on BuildBuddy when master is pushed." >&2
echo "Workflow: buildbuddy.yaml" >&2
echo "Command on the runner: bb build --config=buildbuddy --config=release //:zml_on_arm64" >&2
exit 0
