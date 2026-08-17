#!/usr/bin/env bash
# Phase 1 smoke test: verifies the stable demo-service deployment is
# actually reachable and behaving correctly inside the local cluster --
# not just that `kubectl apply` succeeded. This is the script
# deploy/local-cluster/setup-cluster.sh runs as its final step, and the
# concrete artifact behind Phase 1's exit criterion ("fresh clone can
# deploy locally").
#
# Requires: kubectl (configured against the target cluster), curl.
# Exits non-zero on any failed check so it is safe to use as a CI/script
# gate, not just an interactive sanity check.
set -euo pipefail

export PATH="$HOME/bin:$HOME/go/bin:$PATH"

NAMESPACE="releaseguard"
DEPLOYMENT="demo-service-stable"
SERVICE="svc/demo-service-stable"
POD_SELECTOR="app.kubernetes.io/name=demo-service,app.kubernetes.io/instance=stable"
LOCAL_PORT="18080"
REMOTE_PORT="80"
CONTAINER_PORT="8080"

PF_LOG="$(mktemp)"
BODY_FILE="$(mktemp)"
PF_PID=""

cleanup() {
  if [[ -n "$PF_PID" ]]; then
    kill "$PF_PID" >/dev/null 2>&1 || true
    wait "$PF_PID" 2>/dev/null || true
  fi
  rm -f "$PF_LOG" "$BODY_FILE"
}
trap cleanup EXIT

log() { echo "[smoke] $*"; }

# stop_port_forward + start_port_forward let the script reuse LOCAL_PORT
# across several forward targets in turn (the Service, then each Pod in
# turn) instead of hardcoding one forward for the whole script.
stop_port_forward() {
  if [[ -n "$PF_PID" ]]; then
    kill "$PF_PID" >/dev/null 2>&1 || true
    wait "$PF_PID" 2>/dev/null || true
    PF_PID=""
  fi
}

start_port_forward() {
  local target="$1" remote_port="$2"
  stop_port_forward
  : >"$PF_LOG"
  kubectl -n "$NAMESPACE" port-forward "$target" "${LOCAL_PORT}:${remote_port}" >"$PF_LOG" 2>&1 &
  PF_PID=$!

  local ready=""
  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null 2>&1; then
      ready="1"
      break
    fi
    sleep 0.5
  done
  if [[ -z "$ready" ]]; then
    log "FAIL: port-forward to $target never became reachable; log follows"
    cat "$PF_LOG" >&2
    exit 1
  fi
}

log "waiting for deployment/$DEPLOYMENT rollout"
kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" --timeout=120s

log "starting port-forward 127.0.0.1:${LOCAL_PORT} -> ${SERVICE}:${REMOTE_PORT}"
start_port_forward "$SERVICE" "$REMOTE_PORT"

fail=0

check_status() {
  local desc="$1" url="$2" expect="$3"
  local status
  status="$(curl -s -o "$BODY_FILE" -w '%{http_code}' "$url")"
  if [[ "$status" != "$expect" ]]; then
    log "FAIL: $desc -- expected HTTP $expect, got $status. Body: $(cat "$BODY_FILE")"
    fail=1
  else
    log "PASS: $desc ($status)"
  fi
}

check_contains() {
  local desc="$1" url="$2" needle="$3"
  local body
  body="$(curl -sf "$url")"
  if [[ "$body" == *"$needle"* ]]; then
    log "PASS: $desc"
  else
    log "FAIL: $desc -- expected response to contain '$needle', got: $body"
    fail=1
  fi
}

check_status "GET /health returns 200"  "http://127.0.0.1:${LOCAL_PORT}/health"  200
check_status "GET /version returns 200" "http://127.0.0.1:${LOCAL_PORT}/version" 200
check_status "GET /work returns 200"    "http://127.0.0.1:${LOCAL_PORT}/work"    200
check_status "GET /metrics returns 200" "http://127.0.0.1:${LOCAL_PORT}/metrics" 200

check_contains "/version reports track=stable" "http://127.0.0.1:${LOCAL_PORT}/version" '"track":"stable"'
check_contains "/metrics exposes http_requests_total"           "http://127.0.0.1:${LOCAL_PORT}/metrics" "http_requests_total"
check_contains "/metrics exposes http_request_duration_seconds" "http://127.0.0.1:${LOCAL_PORT}/metrics" "http_request_duration_seconds"

# Replica-count sanity check first: cheap, and catches a deployment that
# never reached its desired size before we spend time port-forwarding to
# individual pods below.
ready_replicas="$(kubectl -n "$NAMESPACE" get "deployment/$DEPLOYMENT" -o jsonpath='{.status.readyReplicas}')"
desired_replicas="$(kubectl -n "$NAMESPACE" get "deployment/$DEPLOYMENT" -o jsonpath='{.spec.replicas}')"
if [[ "$ready_replicas" != "$desired_replicas" ]]; then
  log "FAIL: readyReplicas=$ready_replicas, want desired=$desired_replicas"
  fail=1
else
  log "PASS: all $desired_replicas replicas ready"
fi

# The Service-level checks above only ever exercised ONE pod: a Service
# port-forward picks a single backing endpoint for the life of the
# forward, so a second/third replica that is individually broken (bad
# config, crash-looping app logic, etc.) would never be exercised and this
# script would still report SMOKE TEST PASSED. Every stable pod is
# therefore forwarded to and checked directly and independently. The demo
# image is distroless (no shell, no curl -- see demo-service/Dockerfile),
# so `kubectl exec` is not an option here; per-pod `kubectl port-forward`
# is the mechanism that actually reaches into each container.
log "checking every stable pod individually (not just via the Service)"
pod_names="$(kubectl -n "$NAMESPACE" get pods -l "$POD_SELECTOR" -o jsonpath='{.items[*].metadata.name}')"
if [[ -z "$pod_names" ]]; then
  log "FAIL: no pods found matching selector '$POD_SELECTOR'"
  fail=1
fi

pod_count=0
for pod in $pod_names; do
  pod_count=$((pod_count + 1))
  log "-- pod/$pod --"
  start_port_forward "pod/${pod}" "$CONTAINER_PORT"

  check_status "pod/$pod GET /health returns 200"  "http://127.0.0.1:${LOCAL_PORT}/health"  200
  check_status "pod/$pod GET /version returns 200" "http://127.0.0.1:${LOCAL_PORT}/version" 200
  check_status "pod/$pod GET /work returns 200"    "http://127.0.0.1:${LOCAL_PORT}/work"    200
  check_status "pod/$pod GET /metrics returns 200" "http://127.0.0.1:${LOCAL_PORT}/metrics" 200
  check_contains "pod/$pod /version reports track=stable" "http://127.0.0.1:${LOCAL_PORT}/version" '"track":"stable"'
done
stop_port_forward

if [[ -n "$desired_replicas" && "$pod_count" != "$desired_replicas" ]]; then
  log "FAIL: checked $pod_count pod(s) individually, want desired=$desired_replicas"
  fail=1
else
  log "PASS: all $pod_count pod(s) individually verified"
fi

if [[ "$fail" -ne 0 ]]; then
  log "SMOKE TEST FAILED"
  exit 1
fi

log "SMOKE TEST PASSED"
