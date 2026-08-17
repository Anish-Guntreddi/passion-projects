# ADR 0008: Terraform cloud deployment is deferred (stretch, not MVP)

**Status:** Accepted
**Decision ID:** D7 (spec §1.9)
**Date:** 2026-08-17

## Context

D7 asks about a Terraform-provisioned cloud deployment target, defaulting
to "stretch, not MVP." The roadmap's stretch-goal list (Part 3) confirms:
"Terraform cloud demo" is listed alongside CRD/operator API, multi-step
rollouts, and other explicitly post-MVP items. Claude Code handoff rule 3
also states the local MVP must stay independent of paid cloud services.

## Decision

No Terraform, no cloud provider account, and no cloud-hosted Kubernetes
cluster are part of this project's MVP (Phases 0-9). Every phase's exit
criterion is achievable against a local kind cluster on the developer's
own machine. If a cloud demo is built later, it is additive (new
`infra/terraform/` or similar, new CI job) and must not become a
prerequisite for any existing phase's exit criterion or for the Definition
of Done, which is scoped to "fresh clone stands up the local environment."

## Consequences

- Zero cloud spend and zero cloud credentials are required to build, test,
  or demo this project end-to-end -- directly testable by any reviewer
  with just Docker + kind + kubectl locally.
- The portfolio narrative (§1.5) and Definition of Done do not depend on
  cloud infrastructure being available or funded, which matters for a
  project meant to be independently reproducible by an interviewer or
  hiring manager.
- If pursued later, the natural shape is a `docs/decisions/000X-cloud-
  target.md` ADR of its own once a concrete provider/target is chosen
  (this ADR only records the *deferral*, not a future design).

## Alternatives considered

- **Building the Terraform stretch goal now, alongside Phase 0/1**:
  rejected outright -- it is explicitly out of scope for the phases this
  change implements, and pulling it forward would violate the "do not
  proceed past a phase" discipline this project (and portfolio) is run
  under.
