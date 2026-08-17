#!/usr/bin/env bash
# Delete the local kind cluster created by setup-cluster.sh. Safe to run
# even if the cluster does not exist (kind exits 0 in that case with a
# "no kind clusters found" style message, which this script does not
# treat as failure).
set -euo pipefail

export PATH="$HOME/bin:$HOME/go/bin:$PATH"

CLUSTER_NAME="releaseguard-local"

if ! command -v kind >/dev/null 2>&1; then
  echo "[teardown-cluster] ERROR: 'kind' not found on PATH." >&2
  exit 1
fi

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  echo "[teardown-cluster] deleting kind cluster '$CLUSTER_NAME'"
  kind delete cluster --name "$CLUSTER_NAME"
else
  echo "[teardown-cluster] kind cluster '$CLUSTER_NAME' does not exist; nothing to do"
fi
