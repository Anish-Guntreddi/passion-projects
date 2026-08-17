#!/usr/bin/env bash
# Phase 3 (Manual canary): flips the running demo-service-canary
# Deployment between a healthy configuration and a deliberately unhealthy
# one, by patching its container env vars (which triggers a rollout, same
# as any Deployment template change) and waiting for that rollout to
# finish. This is the "manual" in "manual canary": Phase 3 has no
# controller yet, so a human decides when the canary should look bad and
# flips it back -- see docs/architecture.md and ADR 0003.
#
# This is deliberately a single healthy/unhealthy toggle, not the full
# failure-injection scenario matrix (latency-only, error-only,
# dependency-down, missing-telemetry, saturation) -- that matrix is
# Phase 7's FR5 deliverable ("Failure-injection demo scripts"). Phase 3's
# exit criterion only requires proving that both healthy and unhealthy
# canary telemetry CAN be produced from the same Deployment via env vars
# alone, with no code change or new image -- this script demonstrates
# exactly that, reusing the fault-injection knobs demo-service/internal/
# config/config.go already validates.
#
# Usage:
#   deploy/local-cluster/canary-scenario.sh healthy
#   deploy/local-cluster/canary-scenario.sh unhealthy
#
# Requires: kubectl (configured against the target cluster), a cluster
# with the canary track already deployed (e.g. via `make cluster-up`).
set -euo pipefail

export PATH="$HOME/bin:$HOME/go/bin:$PATH"

NAMESPACE="releaseguard"
DEPLOYMENT="demo-service-canary"
CONTAINER="demo-service"

log() { echo "[canary-scenario] $*"; }

usage() {
  echo "usage: $0 <healthy|unhealthy>" >&2
  exit 1
}

[[ $# -eq 1 ]] || usage

case "$1" in
  healthy)
    ERROR_RATE="0.0"
    EXTRA_LATENCY_MS="0"
    DEPENDENCY_DOWN="false"
    ;;
  unhealthy)
    # Elevated error rate *and* a latency regression together, so a human
    # (or, from Phase 4 on, the evaluator) has two independent signals to
    # call this canary unhealthy, not just one borderline metric.
    ERROR_RATE="0.5"
    EXTRA_LATENCY_MS="250"
    DEPENDENCY_DOWN="false"
    ;;
  *)
    usage
    ;;
esac

log "setting demo-service-canary to '$1' (ERROR_RATE=$ERROR_RATE EXTRA_LATENCY_MS=$EXTRA_LATENCY_MS DEPENDENCY_DOWN=$DEPENDENCY_DOWN)"
kubectl -n "$NAMESPACE" set env "deployment/$DEPLOYMENT" -c "$CONTAINER" \
  "ERROR_RATE=$ERROR_RATE" \
  "EXTRA_LATENCY_MS=$EXTRA_LATENCY_MS" \
  "DEPENDENCY_DOWN=$DEPENDENCY_DOWN"

log "waiting for deployment/$DEPLOYMENT rollout"
kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" --timeout=120s

log "done -- demo-service-canary is now '$1'. Verify via:"
log "  kubectl -n $NAMESPACE port-forward svc/demo-service-canary 18081:80"
log "  curl -s localhost:18081/health | jq ."
log "  curl -s localhost:18081/work -o /dev/null -w '%{http_code}\n'"
