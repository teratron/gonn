---
phase: 4
name: "Examples Catalog (Track D)"
status: Blocked
subsystem: "examples/"
requires:
  - "phase-2: pkg/nn public facade"
  - "phase-3: pkg/persistence, pkg/checkpoint, pkg/dataset, pkg/compute, pkg/perf"
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 4 Tasks — Examples Catalog (Track D)

**Phase:** 4
**Status:** `Blocked`
**Strategic Goal:** Smoke-test catalog validating every track end-to-end. 15 canonical examples per `l2-usage-examples` v1.0.0 — covers the full feature matrix from the simplest XOR network through persistence round-trip, mid-training checkpointing, and the streaming dataset path.

## Block Reasoning

Phase 4 depends transitively on **both** Phase 2 (public facade — examples consume `pkg/nn.NN[T]`) and Phase 3 (capability packages — half of the catalog exercises persistence, checkpointing, streaming, or the CPU backend). Atomic task decomposition deferred until both phases reach `Done`.

## Provisional Catalog (15 entries)

Recorded for forward reference; **not yet decomposed into atomic tasks**. The canonical list is owned by [l2-usage-examples.md](../specifications/l2-usage-examples.md); this stub mirrors the count only.

| Group | Count | Theme |
| :--- | :--- | :--- |
| Builder API basics | 3-4 | XOR / regression / classification — fluent chain |
| Options API basics | 2-3 | Same nets via functional options + presets |
| Training control | 2 | Pause/Resume, early stopping via LossLimit |
| Persistence | 2 | Save/Load network config + weights |
| Checkpointing | 2 | Snapshot mid-training + recovery |
| Streaming dataset | 1-2 | Bounded-memory feed |
| Performance | 1 | pprof + sync.Pool integration |

## Unblock Procedure

1. Phases 2 and 3 both reach `Done`.
2. Run `magic.task update` — atomic tasks `T-4XNN` will be generated per catalog entry.
3. Run `magic.run` to execute.
