---
phase: 3
name: "New Capability Packages (Track C)"
status: Blocked
subsystem: "pkg/persistence, pkg/checkpoint, pkg/dataset, pkg/compute, pkg/perf"
requires:
  - "phase-1: pkg/utils, pkg/neuron, pkg/layer, pkg/network"
  - "L1 parents must reach Stable: l1-network-persistence, l1-checkpointing, l1-data-streaming, l1-compute-backend, l1-performance-contract"
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 3 Tasks — New Capability Packages (Track C)

**Phase:** 3
**Status:** `Blocked`
**Strategic Goal:** Add `pkg/persistence`, `pkg/checkpoint`, `pkg/dataset`, `pkg/compute/cpu`, and the performance harness — capabilities not present in the legacy code. Independent of Phase 2 — runs in parallel once Phase 1 is green and the L1 parent specs reach Stable.

## Block Reasoning (C12 Quarantine)

All five L2 implementation specs in this phase have RFC-status L1 parents:

| L2 Spec | L1 Parent | Parent Status |
| :--- | :--- | :--- |
| `l2-persistence-impl` | `l1-network-persistence` | RFC v0.1.0 |
| `l2-checkpointing-impl` | `l1-checkpointing` | RFC v0.1.0 |
| `l2-streaming-impl` | `l1-data-streaming` | RFC v0.1.0 |
| `l2-backend-cpu` | `l1-compute-backend` | RFC v0.1.0 |
| `l2-perf-impl` | `l1-performance-contract` | RFC v0.1.0 |

Per `RULES.md` C12 (Quarantine Cascade), an L2 spec cannot promote to active planning while its L1 parent is non-Stable. Phase 3 stays Blocked until at least one L1 parent reaches Stable via `magic.spec`. Atomic task decomposition deferred to that point.

## Unblock Procedure

1. Run `magic.spec` on the relevant L1 parent(s) to drive RFC → Stable.
2. Run `magic.task update` to refresh this phase: tasks will be generated track-by-track for each unblocked L2 spec.
3. Run `magic.run` to begin execution.

## Provisional Track Layout

Recorded for forward reference; **not yet decomposed into atomic tasks**.

| Track | Scope | Primary Spec |
| :--- | :--- | :--- |
| A — Persistence | `pkg/persistence` JSON round-trip | l2-persistence-impl |
| B — Checkpointing | `pkg/checkpoint` snapshot/recovery | l2-checkpointing-impl |
| C — Streaming | `pkg/dataset` bounded-memory iterator | l2-streaming-impl |
| D — CPU Backend | `pkg/compute/cpu` reference path | l2-backend-cpu |
| E — Performance | sync.Pool + pprof harness | l2-perf-impl |
