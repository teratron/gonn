# Master Task Index (Registry)

**Version:** 1.5.0
**Project Version:** 0.1.0 (initial release target — semver baseline)
**Generated:** 2026-04-29
**Last Updated:** 2026-05-02
**Based on:** .design/PLAN.md v1.5.0
**Based on RULES:** .design/RULES.md v1.2.0
**Execution Mode:** Parallel (C3)
**Status:** Active

## Overview

Tactical registry of all phases and their statuses. Atomic checklists (`T-XXXX`) live in per-phase files inside `tasks/`. Progress (`[x]`, `[/]`, `[!]`) is recorded in those files first; this index summarizes phase-level status only.

## Active Phases

| Phase | Description | Status |
| :--- | :--- | :--- |
| [Phase 1](tasks/phase-1.md) | Foundation Rewrite (Track A) — `pkg/utils/{errors,init}`, `pkg/neuron`, `pkg/layer`, `pkg/network` | `Done` (2026-04-29) |
| [Phase 2](tasks/phase-2.md) | Public Facade Restoration (Track B) — `pkg/nn` | `Done` (2026-04-30) |
| [Phase 3](tasks/phase-3.md) | New Capability Packages (Track C) — persistence, checkpoint, dataset, compute, perf | `Done` (2026-05-01) |
| [Phase 4](tasks/phase-4.md) | Examples Catalog (Track D) — `examples/E01..E15` (v0.1 scope: 7 entries) | `Active` (decomposed 2026-05-02; 14 tasks + 4 gate checks) |

## Phase 0 — Already Complete

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/activation/` | l2-activation-functions v1.0.0 | `Stable` |
| `pkg/loss/` | l2-loss-functions v1.0.0 | `Stable` |
| `pkg/utils/float.go`, `pkg/utils/logger.go` | utils baseline | `Stable` |

## Phase 1 Deliverables (Done)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/utils/errors.go` | l2-errors-impl v1.0.0 | `Stable` |
| `pkg/utils/init.go` | l2-init-impl v1.0.0 | `Stable` |
| `pkg/neuron/cell/*` | l2-neuron-model v1.1.0 | `Stable` |
| `pkg/neuron/axon/*` | l2-neuron-model v1.1.0 | `Stable` |
| `pkg/layer/*` | l2-layer-types v1.1.0 | `Stable` |
| `pkg/network/*` | l2-network-graph v1.1.0 | `Stable` |

## Phase 2 Deliverables (Done)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/nn/{config,nn,builder,compile}.go` | l2-nn-facade v2.0.0 | `Stable` |
| `pkg/nn/{options,presets}.go` | l2-nn-facade v2.0.0 | `Stable` |
| `pkg/nn/{train,query,verify}.go` | l2-training-loop v1.0.0 | `Stable` |
| `pkg/nn/control.go` | l2-control-impl v1.0.0 | `Stable` |

## Phase 3 Deliverables (Done)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/persistence/{config,weights}.go` | l2-persistence-impl v1.0.0 | `Stable` |
| `pkg/checkpoint/{snapshot,writer,reader,retention}.go` | l2-checkpointing-impl v1.0.0 | `Stable` |
| `pkg/dataset/{dataset,slice,csv,prefetch}.go` | l2-streaming-impl v1.0.0 | `Stable` |
| `pkg/compute/{backend,registry}.go` + `cpu/*` | l2-backend-cpu v1.0.0 | `Stable` |
| `pkg/network/{pool,worker}.go` + `pkg/nn/profiling.go` + bench files | l2-perf-impl v1.0.0 | `Stable` |

## Phase 4 Deliverables (Pending — v0.1 scope)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `examples/xor/` (E01) | l2-usage-examples v1.0.0 | `Pending` |
| `examples/style_showcase/` (E12) | l2-usage-examples v1.0.0 | `Pending` |
| `examples/logic_gates/` (E02) | l2-usage-examples v1.0.0 | `Pending` |
| `examples/callbacks/` (E11) | l2-usage-examples v1.0.0 | `Pending` |
| `examples/persistence/` (E09) | l2-usage-examples v1.0.0 + l2-persistence-impl v1.0.0 | `Pending` |
| `examples/shared_options/` (E14, adapted) | l2-usage-examples v1.0.0 | `Pending` |
| `examples/precision/` (E15) | l2-usage-examples v1.0.0 | `Pending` |
| `examples/README.md` + coverage audit | l2-usage-examples v1.0.0 §5.3 | `Pending` |

### Phase 4 v0.2 Backlog (8 entries — multi-hidden / AndTrain / MNIST)

E03, E04, E05, E06, E07, E08, E10, E13 — wait on v0.2 multi-hidden (`pkg/nn.compile()` accepts `len(HiddenLayers) > 1`), `AndTrain` API surface, or MNIST dataset-loader spec.

## Cross-Cutting Constraints

- **C29 Stdlib-only**: every new file uses Go standard library exclusively.
- **C30 Coverage floor**: every new package ships with ≥80% line coverage and `go test -race` clean.
- **C31 Doc-comments**: tier-by-audience verbosity (public > internal > unexported).
- **C32 Error informativeness**: every new error wraps a category sentinel from `pkg/utils/errors.go` (introduced in T-1A01).
- **C-001 Resolution**: Phase 1 closes the compilation blocker recorded in STATE.md. No integration testing until `go build ./...` is green.

## Meta Information

- **Last Updated**: 2026-05-02
- **Maintainer**: Core Team
