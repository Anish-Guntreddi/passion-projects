#!/usr/bin/env bash
# Phase 3 (Manual canary) smoke test: verifies the demo-service-canary
# Deployment is reachable and correctly self-identifies (track=canary,
# a distinct version from stable), AND that the shared `demo-service`
# Service (ADR 0003's traffic-split mechanism) actually load-balances
# across BOTH tracks -- i.e. that "controlled split/routing" is real,
# wired-up behavior and not just declared YAML.
#
# Complements tests/e2e/smoke_test.sh (Phase 1, stable track only). Run
# after deploy/local-cluster/setup-cluster.sh / `make cluster-up`, which
# deploys both tracks.
#
# Requires: kubectl (configured against the target cluster), curl.
# Exits non-zero on any failed check so it is safe to use as a CI/script
# gate, not just an interactive sanity check.
set -euo pipefail

export PATH="$HOME/bin:$HOME/go/bin:$PATH"

NAMESPACE="releaseguard"
DEPLOYMENT="demo-service-canary"
SERVICE="svc/demo-service-canary"
SHARED_SERVICE="demo-service"
STABLE_POD_SELECTOR="app.kubernetes.io/name=demo-service,app.kubernetes.io/instance=stable"
CANARY_POD_SELECTOR="app.kubernetes.io/name=demo-service,app.kubernetes.io/instance=canary"
LOCAL_PORT="18081"
REMOTE_PORT="80"
CONTAINER_PORT="8080"
EXPECT_VERSION="v1.1.0"

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

log() { echo "[canary-smoke] $*"; }

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

log "waiting for deployment/$DEPLOYMENT rollout"
kubectl -n "$NAMESPACE" rollout status "deployment/$DEPLOYMENT" --timeout=120s

log "starting port-forward 127.0.0.1:${LOCAL_PORT} -> ${SERVICE}:${REMOTE_PORT}"
start_port_forward "$SERVICE" "$REMOTE_PORT"

check_status "GET /health returns 200"  "http://127.0.0.1:${LOCAL_PORT}/health"  200
check_status "GET /version returns 200" "http://127.0.0.1:${LOCAL_PORT}/version" 200
check_status "GET /work returns 200"    "http://127.0.0.1:${LOCAL_PORT}/work"    200
check_status "GET /metrics returns 200" "http://127.0.0.1:${LOCAL_PORT}/metrics" 200

check_contains "/version reports track=canary" "http://127.0.0.1:${LOCAL_PORT}/version" '"track":"canary"'
check_contains "/version reports release_version=$EXPECT_VERSION" "http://127.0.0.1:${LOCAL_PORT}/version" "\"release_version\":\"${EXPECT_VERSION}\""
check_contains "/metrics exposes http_requests_total"           "http://127.0.0.1:${LOCAL_PORT}/metrics" "http_requests_total"
check_contains "/metrics exposes http_request_duration_seconds" "http://127.0.0.1:${LOCAL_PORT}/metrics" "http_request_duration_seconds"

ready_replicas="$(kubectl -n "$NAMESPACE" get "deployment/$DEPLOYMENT" -o jsonpath='{.status.readyReplicas}')"
desired_replicas="$(kubectl -n "$NAMESPACE" get "deployment/$DEPLOYMENT" -o jsonpath='{.spec.replicas}')"
if [[ "$ready_replicas" != "$desired_replicas" ]]; then
  log "FAIL: readyReplicas=$ready_replicas, want desired=$desired_replicas"
  fail=1
else
  log "PASS: all $desired_replicas canary replica(s) ready"
fi

log "checking every canary pod individually (not just via the Service)"
pod_names="$(kubectl -n "$NAMESPACE" get pods -l "$CANARY_POD_SELECTOR" -o jsonpath='{.items[*].metadata.name}')"
if [[ -z "$pod_names" ]]; then
  log "FAIL: no pods found matching selector '$CANARY_POD_SELECTOR'"
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
  check_contains "pod/$pod /version reports track=canary" "http://127.0.0.1:${LOCAL_PORT}/version" '"track":"canary"'
done
stop_port_forward

if [[ -n "$desired_replicas" && "$pod_count" != "$desired_replicas" ]]; then
  log "FAIL: checked $pod_count pod(s) individually, want desired=$desired_replicas"
  fail=1
else
  log "PASS: all $pod_count canary pod(s) individually verified"
fi

# Split-routing check (ADR 0003): the shared `demo-service` Service's
# Endpoints must include pod IPs from BOTH tracks. A Service port-forward
# only ever reaches one backing pod for the life of the forward (see
# tests/e2e/smoke_test.sh's comment on the same limitation), so it cannot
# prove load-balancing by itself -- the Endpoints object is what actually
# proves both tracks are wired into the one Service kube-proxy balances
# across, independent of which single backend any one connection lands on.
log "checking shared Service '$SHARED_SERVICE' endpoints span both tracks"
stable_ips="$(kubectl -n "$NAMESPACE" get pods -l "$STABLE_POD_SELECTOR" -o jsonpath='{.items[*].status.podIP}')"
canary_ips="$(kubectl -n "$NAMESPACE" get pods -l "$CANARY_POD_SELECTOR" -o jsonpath='{.items[*].status.podIP}')"
shared_endpoint_ips="$(kubectl -n "$NAMESPACE" get endpoints "$SHARED_SERVICE" -o jsonpath='{.subsets[*].addresses[*].ip}')"

check_ips_present() {
  local desc="$1" needle_ips="$2" haystack=" $3 "
  if [[ -z "$needle_ips" ]]; then
    log "FAIL: $desc -- no pod IPs found to check against"
    fail=1
    return
  fi
  local ip
  for ip in $needle_ips; do
    case "$haystack" in
      *" $ip "*) ;;
      *)
        log "FAIL: $desc -- IP $ip not found in shared Service endpoints ($3)"
        fail=1
        return
        ;;
    esac
  done
  log "PASS: $desc"
}

check_ips_present "shared Service endpoints include every stable pod IP" "$stable_ips" "$shared_endpoint_ips"
check_ips_present "shared Service endpoints include every canary pod IP" "$canary_ips" "$shared_endpoint_ips"

if [[ "$fail" -ne 0 ]]; then
  log "CANARY SMOKE TEST FAILED"
  exit 1
fi

log "CANARY SMOKE TEST PASSED"
