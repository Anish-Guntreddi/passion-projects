#!/usr/bin/env bash
# Stand up (or reuse) a local kind cluster, build the demo-service image,
# load it into the cluster, apply the local Kustomize overlay, wait for
# the rollout, and run the smoke test.
#
# This is the single command Phase 1's exit criterion ("fresh clone can
# deploy locally") is checked against. It is intentionally idempotent:
# running it twice does not create a second cluster or duplicate
# resources -- `kubectl apply` is itself idempotent, and cluster/image
# creation are both guarded by existence checks.
#
# Usage:
#   deploy/local-cluster/setup-cluster.sh
#
# Requires on PATH: docker, kind, kubectl.
set -euo pipefail

# Defensively include common user-local install locations (e.g. `go
# install`'s default GOBIN, or a manual `~/bin` install) so this script
# works regardless of whether the caller's shell profile already does --
# harmless no-op if kind/kubectl are already found earlier on PATH.
export PATH="$HOME/bin:$HOME/go/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CLUSTER_NAME="releaseguard-local"
IMAGE_TAG="releaseguard/demo-service:local"
NAMESPACE="releaseguard"
DEPLOYMENT="demo-service-stable"

log() { echo "[setup-cluster] $*"; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[setup-cluster] ERROR: '$1' not found on PATH. See projects/08-releaseguard/README.md for prerequisites." >&2
    exit 1
  fi
}

require_cmd docker
require_cmd kind
require_cmd kubectl

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  log "kind cluster '$CLUSTER_NAME' already exists; reusing it"
else
  log "creating kind cluster '$CLUSTER_NAME'"
  kind create cluster --name "$CLUSTER_NAME" --config "$SCRIPT_DIR/kind-config.yaml"
fi

# `kind create cluster` sets kubectl's current-context for us, but that
# only happens on the create branch above. On the reuse branch nothing
# touches the current-context, so if the caller's kubeconfig had since been
# pointed at some other cluster (a different kind cluster, minikube, or a
# real one) every kubectl/kubectl-apply call below would silently target
# the wrong cluster instead of failing loudly. Setting the context
# explicitly here, unconditionally, makes both branches converge on the
# same guaranteed state.
log "setting kubectl context to 'kind-${CLUSTER_NAME}'"
kubectl config use-context "kind-${CLUSTER_NAME}" >/dev/null

GIT_COMMIT="local"
if git -C "$PROJECT_ROOT" rev-parse --short HEAD >/dev/null 2>&1; then
  GIT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)"
fi
BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

log "building demo-service image ($IMAGE_TAG, commit=$GIT_COMMIT)"
docker build \
  --build-arg VERSION=local \
  --build-arg GIT_COMMIT="$GIT_COMMIT" \
  --build-arg BUILD_TIME="$BUILD_TIME" \
  -t "$IMAGE_TAG" \
  "$PROJECT_ROOT/demo-service"

log "loading image into kind cluster"
kind load docker-image "$IMAGE_TAG" --name "$CLUSTER_NAME"

log "applying manifests (deploy/kubernetes/overlays/local)"
kubectl apply -k "$PROJECT_ROOT/deploy/kubernetes/overlays/local"

# The image tag is a mutable ":local" pointer (see the overlay's comment),
# and `kind load docker-image` above always refreshes what that tag points
# to in the node's containerd -- but imagePullPolicy: IfNotPresent means
# kubelet only ever consults the tag when a Pod is (re)created. On a
# re-run against an already-deployed stable track, `kubectl apply` above
# is a no-diff apply (same image reference, same replica count), so it
# does not itself recreate any Pods -- already-running Pods would keep
# executing the *previous* build's binary even though a newer image was
# just loaded onto the node, and the smoke test below would pass against
# stale code. Forcing a rollout restart makes every re-run pick up the
# freshly-built image; on the very first run (no Pods exist yet) it is a
# harmless no-op layered onto the initial rollout.
log "restarting deployment/$DEPLOYMENT so it picks up the freshly-loaded image"
kubectl -n "$NAMESPACE" rollout restart "deployment/$DEPLOYMENT"

log "waiting for deployment/$DEPLOYMENT rollout"
kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" --timeout=120s

log "running smoke test"
"$PROJECT_ROOT/tests/e2e/smoke_test.sh"

log "done -- stable track is deployed and passed its smoke test"
