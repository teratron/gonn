---
phase: 9
name: Metric Schedulers + Dynamic Topology + Observability Stack
status: Done
subsystem: pkg/optimizer, pkg/network, pkg/utils, pkg/visualization, pkg/nn
requires:
  - phase-8 (ExponentialLR + CLI binary complete; v0.7.0 tagged)
provides:
  - MetricScheduler[T] interface + ReduceOnPlateau[T] + OneCycleLR[T]
  - Dynamic topology mutations (AddNeuron/RemoveNeuron/AddHiddenLayer/RemoveHiddenLayer)
  - HTTP observability server (pkg/visualization)
  - GoLogger slog adapter (pkg/utils)
  - v0.8.0 tag
key_files:
  created:
    - pkg/optimizer/metric_scheduler.go
    - pkg/optimizer/reduce_on_plateau.go
    - pkg/optimizer/one_cycle_lr.go
    - pkg/network/topology.go
    - pkg/visualization/server.go
    - pkg/visualization/handlers.go
    - pkg/visualization/middleware.go
    - pkg/nn/topology.go
  modified:
    - pkg/optimizer/scheduler.go
    - pkg/utils/errors.go
    - pkg/utils/logger.go
    - pkg/network/network.go
    - pkg/nn/train.go
    - pkg/nn/compile.go
    - pkg/nn/config.go
    - pkg/nn/options.go
    - pkg/nn/builder.go
    - pkg/nn/nn.go
    - pkg/checkpoint/snapshot.go
patterns_established:
  - MetricScheduler[T] forward via boundScheduler type-assertion
  - topologyTx[T] rollback pattern for atomic topology mutations
  - VisServer SnapFn callback (pull-based, no locking on NN)
duration_minutes: ~
---

# Phase 9 — Metric Schedulers + Dynamic Topology + Observability Stack

**Status:** Done
**Decomposed:** 2026-05-10
**Completed:** 2026-05-10
**Tasks:** 11 feature + 3 validation + 1 gate = 15 total
**Specs:** l2-metric-scheduler-impl v0.1.0, l2-dynamic-topology-impl v0.2.0, l2-visualization-api v0.2.0, l2-logging-strategy v0.2.0
**Track order:** A, B, C fully parallel (no shared write paths);
                 T-9T01 after Track A; T-9T02 after Track B; T-9T03 after Track C;
                 Gate T-9Z01 after all tracks.

## Track A — Metric Schedulers (pkg/optimizer/)

*Goal: Implement MetricScheduler[T] interface + ReduceOnPlateau[T] + OneCycleLR[T] from
l2-metric-scheduler-impl.md. Closes the Phase 8 backlog item "require MetricScheduler interface
design decision".*
*Source: [l2-metric-scheduler-impl.md](../specifications/l2-metric-scheduler-impl.md)*

- [x] **T-9A01** — Create `pkg/optimizer/metric_scheduler.go`: `MetricScheduler[T]` interface
  embedding `Scheduler[T]` + `StepWithMetric(metric T) T`. Compile-time assertions for both
  new types. No implementation logic in this file — interface only.

- [x] **T-9A02** — Create `pkg/optimizer/reduce_on_plateau.go` + `reduce_on_plateau_test.go`:
  `ReduceOnPlateau[T]` — `patience`/`factor`/`threshold`/`minLR`/`mode` fields; functional options
  constructor `NewReduceOnPlateau[T]`; `Step() T` delegates to `StepWithMetric(0)`;
  `StepWithMetric(metric T) T` implements patience algorithm per spec §5.3; `Granularity()` = PerEpoch;
  `Reset()`, `SaveState()`/`LoadState()`, `reducePlateauState[T]` JSON struct.
  Tests: patience trigger, mode="max", Reset(), SaveState/LoadState round-trip, no-op Step().

- [x] **T-9A03** — Create `pkg/optimizer/one_cycle_lr.go` + `one_cycle_lr_test.go`:
  `OneCycleLR[T]` — 3-phase schedule per spec §5.4 (warm-up → cosine decay → final anneal);
  `NewOneCycleLR[T]` with functional options (`WithPctStart`, `WithFinalDiv`, `WithDivFactor`);
  `Granularity()` = PerStep; `Step() T` delegates to `StepWithMetric(0)`;
  `StepWithMetric` ignores metric (self-contained curve); `Reset()`, `SaveState()`/`LoadState()`.
  Tests: phase boundaries at warmupSteps and decaySteps, LRS-4 hold after totalSteps,
  SaveState/LoadState round-trip, Granularity default.

- [x] **T-9A04** — Patch `pkg/nn/train.go` epoch-dispatch block: type-assert scheduler to
  `optimizer.MetricScheduler[T]`; if true call `ms.StepWithMetric(T(lastLoss))`; else `sched.Step()`.
  ~5 lines. No change to function signature or public API surface.

## Track B — Dynamic Topology (pkg/network/)

*Goal: Implement TopologyMode opt-in + mutation methods on Network[T] from
l2-dynamic-topology-impl.md §6 phases A→E.*
*Source: [l2-dynamic-topology-impl.md](../specifications/l2-dynamic-topology-impl.md)*

- [x] **T-9B01** — Add to `pkg/utils/errors.go` the 6 new DYN sentinels from spec §5.6:
  `ErrImmutableMode`, `ErrInvalidPosition`, `ErrImmutableLayer`, `ErrMinimumTopology`,
  `ErrEmptyLayer`, `ErrMutationFailed`. (`ErrInvalidState` reused from control-impl — no addition.)
  Create `pkg/network/topology.go`: `TopologyMode` enum (`Immutable`, `Dynamic`); `topologyTx[T]`
  struct with `Commit`/`Rollback`; `TopologyVersion() uint64` on `Network[T]`; gate checks
  `n.requireDynamic()` and `n.requireIdleOrPaused()`.
  Add `WithTopologyMode[T]` to `pkg/nn/options.go` and `pkg/nn/builder.go` (~2 lines each).
  Run full test suite: `go test ./...` must be green (zero regressions, no mutations yet).

- [x] **T-9B02** — Implement `AddNeuron(layerIdx uint, count uint) error` and
  `RemoveNeuron(layerIdx uint, count uint) error` in `pkg/network/topology.go`.
  Both methods: `requireDynamic()` → `requireIdleOrPaused()` → `topologyTx` begin →
  validate (bounds, min-size) → mutate cell slice → `Rollback` on error / `Commit` on success →
  increment `TopologyVersion`. Rebalance axons between mutated layer and its neighbours using
  `rebalance()` helper (init strategy from `net.config.WeightInit`).
  Begin `pkg/network/topology_test.go` (TDD from this task onward): AddNeuron, RemoveNeuron,
  error paths, Rollback on validation failure.

- [x] **T-9B03** — Implement `AddHiddenLayer(position uint, size uint, activation activation.Type, bias bool) error`
  and `RemoveHiddenLayer(position uint) error` in `pkg/network/topology.go`.
  `AddHiddenLayer`: insert new Dense layer at position in hidden slice; rebalance predecessor
  and successor axons. `RemoveHiddenLayer`: check `len(hidden) > 1` (ErrMinimumTopology);
  remove layer; bridge re-wire predecessor→successor. Both increment `TopologyVersion`.
  Extend `topology_test.go`: AddHiddenLayer mid-chain, RemoveHiddenLayer, ErrMinimumTopology guard,
  immutable mode rejection, invalid position out-of-range.

- [x] **T-9B04** — Wire `TopologyVersion` into observability and checkpointing:
  Expose `TopologyVersion() uint64` via `pkg/nn/nn.go` (delegation to `n.Network.TopologyVersion()`).
  Add `TopologyVersion uint64` field to `pkg/checkpoint/` snapshot JSON schema
  (`pkg/checkpoint/writer.go` and `reader.go`). Update `topology_test.go` with checkpoint round-trip.

## Track C — Observability Stack (pkg/utils/ + pkg/visualization/ + pkg/nn/)

*Goal: Upgrade pkg/utils/logger.go to log/slog; create pkg/visualization/ HTTP server and handlers.
Wire both via pkg/nn/ options.*
*Source: [l2-logging-strategy.md](../specifications/l2-logging-strategy.md) §5.4–5.5*
*Source: [l2-visualization-api.md](../specifications/l2-visualization-api.md) §5.4–5.6*
*Note: T-9C01 → T-9C03 intra-track dependency (slog wrapper before nn wiring).*

- [x] **T-9C01** — Refactor `pkg/utils/logger.go`: define `LevelTrace = slog.Level(-8)`;
  replace opaque `Logger` shim with `goLogger` struct wrapping `*slog.Logger` (spec §5.5);
  inject `lib_version` + `network_id` fields at construction via `slog.With`;
  provide `Trace/Debug/Info/Warn/Error` methods forwarding to slog at corresponding levels;
  fallback discard handler when `nil` logger provided.
  Update `pkg/utils/logger_test.go` (or create): LevelTrace constant, required fields injection,
  nil-safe fallback.

- [x] **T-9C02** — Create `pkg/visualization/` package (new directory):
  `server.go`: `VisServer` struct (spec §5.5), `NewVisServer`, `RegisterNetwork`, `Start`, `Stop`.
  `handlers.go`: `snapshotHandler`, `lossHandler`, `activationsHandler`, `controlHandler`,
  `statsHandler`, `healthHandler` — all emit `"protocol_version":"1.0.0"` JSON (spec §5.6).
  `middleware.go`: bearer-token auth middleware (no-op when token empty).
  `server_test.go`: `httptest.NewServer`-based tests for each endpoint (200 OK, 404 on bad layerIdx,
  401 on missing token when token configured). Coverage ≥80 %.

- [x] **T-9C03** — Wire observability options into `pkg/nn/`:
  `pkg/nn/options.go`: add `WithLogger[T](*slog.Logger)`, `WithVisualizationEndpoint[T](addr string)`,
  `WithVisualizationToken[T](token string)`, `WithVisualizationCORS[T](bool)`.
  `pkg/nn/config.go`: add `Logger *slog.Logger`, `VisAddr string`, `VisToken string`, `VisCORS bool`.
  `pkg/nn/compile.go`: after compile() success, if `VisAddr != ""` start `VisServer` and store
  reference on NN struct.
  `pkg/nn/nn.go`: add `Close() error` method that stops VisServer if running.
  Add training lifecycle log events in `pkg/nn/train.go` (Info: start/stop; Debug: epoch; Trace: iter).

## Validation Tasks

- [x] **T-9T01** — Metric schedulers validation (after Track A):
  - `go test ./pkg/optimizer/...` — all scheduler tests green (including new types).
  - `ReduceOnPlateau` coverage ≥80 %; `OneCycleLR` coverage ≥80 %; overall `pkg/optimizer/` ≥80 %.
  - Verify `SaveState`/`LoadState` round-trip produces bit-identical rate outputs.
  - Verify `BindScheduler` wires correctly to SGD/Adam/RMSProp/Momentum for both new types.
  - Update `l2-lr-scheduling-impl.md` §5.3 Deferred list: move ReduceOnPlateau and OneCycleLR
    to Implemented. Bump spec to v1.2.0. Update INDEX.md entry.

- [x] **T-9T02** — Dynamic topology validation (after Track B):
  - `go test -count=1 ./pkg/network/...` green; `topology_test.go` ≥80 % coverage.
  - Verify Immutable network rejects all mutation methods with `ErrImmutableMode`.
  - Verify `topologyTx.Rollback()` restores layer slice on validation error (snapshot integrity).
  - Verify `TopologyVersion` increments per commit.
  - Verify `go build ./...` clean (no regression on existing network code).

- [x] **T-9T03** — Observability stack validation (after Track C):
  - `go test -count=1 ./pkg/visualization/...` green; coverage ≥80 %.
  - `go test -count=1 ./pkg/utils/...` green; LevelTrace and goLogger covered.
  - End-to-end smoke: build XOR net with `WithVisualizationEndpoint(":0")` → `GET /v1/snapshot`
    returns valid JSON with `protocol_version`.
  - Verify `Close()` stops the HTTP server (no port leak).
  - Verify logger discard fallback: constructing NN with `WithLogger(nil)` does not panic.

## Gate

- [x] **T-9Z01** — Phase 9 gate:
  - `go build ./...` clean.
  - `go test -count=1 ./...` all green (no regressions from prior phases).
  - Every new / modified package ≥80 % line coverage:
    `pkg/optimizer/` (with new metric types), `pkg/network/` (topology), `pkg/visualization/` (new),
    `pkg/utils/` (logger).
  - `l2-lr-scheduling-impl.md` updated to v1.2.0 (ReduceOnPlateau + OneCycleLR in Implemented list).
  - `CHANGELOG.md` v0.8.0 entry written.
  - `v0.8.0` git tag created per `l1-release-policy.md §5.4`.
