# ADR 0001: Controller (and demo-service) language is Go

**Status:** Accepted
**Decision ID:** D1 (spec §1.9)
**Date:** 2026-08-17

## Context

The spec's open decision D1 asks Go vs. Python for the ReleaseGuard
controller, with a recommended default of Go "for stronger
Kubernetes/controller signal." Separately, the tech-stack table allows the
demo-service to be written in any language "keep simple... one language
reduces toil."

## Decision

The controller (Phase 4+) is written in Go, per the spec's default. The
demo-service (Phase 0, this change) is also written in Go, so the whole
repository is a single-language, single-toolchain project.

Concretely, `demo-service` and the future `controller` are two independent
Go modules (`demo-service/go.mod`, later `controller/go.mod`) rather than
one shared module. They are conceptually separate deployables with
different dependency footprints -- the demo-service only needs an HTTP
server and a Prometheus client; the controller will need `client-go`,
a YAML/JSON schema validator, and a Prometheus HTTP query client. Keeping
them as separate modules means neither pulls in the other's dependencies,
and each can be versioned/released independently later if this repo is
ever split.

## Consequences

- One toolchain (Go 1.22) for CI, local dev, and containers across the
  whole project -- no polyglot tooling to maintain for a portfolio piece
  whose main point is the controller/evidence pipeline, not the demo app.
- `client-go` (used starting Phase 6, the Kubernetes action adapter) is a
  natural fit for a Go controller and is the same library real Kubernetes
  controllers/operators use, which is directly the "stronger
  Kubernetes/controller signal" the spec calls out.
- Two `go.mod` files means `go test ./...` from the repo root does not
  work; CI and local tooling must `cd` into each module (see
  `Makefile` and `.github/workflows/ci.yml`, which both set
  `working-directory: demo-service` explicitly).

## Alternatives considered

- **Python controller**: better fit for teams already invested in
  Python/asyncio, and viable via the `kubernetes` PyPI client, but weaker
  signal for a controller-pattern portfolio piece and the spec's own
  default already picks Go.
- **Single shared Go module for controller + demo-service**: simpler
  `go.work`-free setup, rejected because it would force the demo-service
  binary's container image to build (and vendor) controller dependencies
  (client-go, etc.) it never uses, bloating the image and blurring the
  "demo app is not the accomplishment" boundary the spec is explicit about
  (§1.1: "the important implementation is the decision/controller logic
  ... not a collection of YAML files").
