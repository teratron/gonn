# Master Task Index (Registry)

**Version:** 1.8.0
**Project Version:** 0.5.1 (v0.5 released; v0.6 complete; Phase 6 active)
**Generated:** 2026-04-29
**Last Updated:** 2026-05-07
**Based on:** .design/PLAN.md v1.8.0
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
| [Phase 4](tasks/phase-4.md) | Examples Catalog (Track D) — `examples/E01..E15` (v0.5 scope: 7 entries) | `Done` (2026-05-02) |
| [Phase 5](tasks/phase-5.md) | Multi-Hidden Topology (v0.6) — `pkg/network` chain + compile() lift + weights 1.1.0 + 6 catalog examples | `Done` (2026-05-06) |
| [Phase 6](tasks/phase-6.md) | Feature Expansion + v0.6 Release — `pkg/optimizer/`, `pkg/regularizer/`, axon WeightInit fix, CHANGELOG, v0.6.0 tag | `Active` (decomposed 2026-05-07; 15 tasks + 2 validation + 2 gate) |

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

## Phase 4 Deliverables (Done — v0.5 scope)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `examples/xor/` (E01) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/style_showcase/` (E12) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/logic_gates/` (E02) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/callbacks/` (E11) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/persistence/` (E09) | l2-usage-examples v1.0.0 + l2-persistence-impl v1.0.0 | `Stable` |
| `examples/shared_options/` (E14, adapted) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/precision/` (E15) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/README.md` + coverage audit | l2-usage-examples v1.0.0 §5.3 | `Stable` |

### Phase 4 v0.6 Promotion → Phase 5

Six of the eight v0.6-gated entries (E03, E04, E05, E07, E08, E13) are now active under Phase 5 — see deliverables table below.

## Phase 5 Deliverables (Done — 2026-05-06)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/network/{network,propagation}.go` (multi-hidden chain) | l2-multihidden-impl v1.0.0 | `Done` |
| `pkg/nn/compile.go` (gate lift + variadic SetLayers) | l2-multihidden-impl v1.0.0 | `Done` |
| `pkg/persistence/config.go` (SchemaVersion 1.1.0) | l2-multihidden-impl v1.0.0 §5.4 | `Done` |
| `examples/perceptron/` restored (E03) | l2-usage-examples v1.0.0 | `Done` |
| `examples/binary_classification/` (E04) | l2-usage-examples v1.0.0 | `Done` |
| `examples/iris/` (E05) | l2-usage-examples v1.0.0 | `Done` |
| `examples/regression_sin/` (E07) | l2-usage-examples v1.0.0 | `Done` |
| `examples/regression_multi/` (E08) | l2-usage-examples v1.0.0 | `Done` |
| `examples/higher_order_options/` (E13) | l2-usage-examples v1.0.0 | `Done` |

## Phase 6 Deliverables (Pending)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/optimizer/` (new package) | l2-optimizer-impl v1.0.0 | `Pending` |
| `pkg/regularizer/` (new package) | l2-regularization-impl v1.0.0 | `Pending` |
| `pkg/neuron/axon/axon.go` (WeightInit debt fix) | l2-regularization-impl v1.0.0 T-6B06 | `Pending` |
| `pkg/nn/options.go` + `compile.go` (WithOptimizer, WithRegularizer) | l2-optimizer-impl + l2-regularization-impl | `Pending` |
| `pkg/nn/train.go` (opt.Step + reg.Penalty + reg.ApplyMask) | l2-training-loop v1.0.0 | `Pending` |
| `CHANGELOG.md` (v0.6.0 entry) | l1-release-policy v1.0.0 | `Pending` |
| `README.md` (v0.6 multi-hidden API docs) | l1-release-policy v1.0.0 | `Pending` |
| `v0.6.0` git tag | l1-release-policy v1.0.0 §5.4 | `Pending` |

### Remaining v0.6 Backlog (gated on follow-up specs)

- **E06 MNIST preset** — awaits MNIST dataset-loader spec (planned post-Phase 5).
- **E10 Continuation** — awaits `AndTrain` API surface (planned post-Phase 5).
- **`PresetRegression` parameter extension** — v0.7 follow-up minor.

## Cross-Cutting Constraints

- **C29 Stdlib-only**: every new file uses Go standard library exclusively.
- **C30 Coverage floor**: every new package ships with ≥80% line coverage and `go test -race` clean.
- **C31 Doc-comments**: tier-by-audience verbosity (public > internal > unexported).
- **C32 Error informativeness**: every new error wraps a category sentinel from `pkg/utils/errors.go` (introduced in T-1A01).
- **C-001 Resolution**: Phase 1 closes the compilation blocker recorded in STATE.md. No integration testing until `go build ./...` is green.

## Meta Information

- **Last Updated**: 2026-05-07
- **Maintainer**: Core Team
