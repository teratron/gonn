# Observability Protocol

**Version:** 0.1.0
**Status:** Draft
**Layer:** concept

## Overview

Defines the **read-only contract** by which external observers (loggers, GUIs, web visualizers,
metrics collectors) inspect a running or trained network without interfering with its operation.
Establishes a uniform set of state-exposure types and access patterns; concrete transport
(callbacks, HTTP, gRPC) is chosen per-implementation in Layer 2 specs.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent
- [l1-training-control.md](l1-training-control.md) — Observability reads control state
- [l2-logging-strategy.md](l2-logging-strategy.md) — Implements observability for log sinks
- [l2-visualization-api.md](l2-visualization-api.md) — Implements observability for external GUIs

## 1. Motivation

Today the only signal is `Train()` returning `(iterations, loss)` once training ends. Production-grade
ML tooling needs continuous read-only access to:

- Current loss / loss history.
- Per-layer activations on the most recent forward pass.
- Per-axon weight histograms.
- Training control state (Running / Paused / etc., per `l1-training-control`).

A separate visualizer repository (planned per TODO #7) consumes this protocol.

## 2. Constraints & Assumptions

- Observers are strictly read-only. Mutating the network from an observer is a contract violation.
- Reads must be **non-blocking**: an observer must never stall the training loop. Use snapshots /
  copies / atomics; never expose a lock that the loop also holds.
- The protocol is **pull-based** at the spec level. Push (callbacks, channels) is a thin Layer 2
  adapter over the pull contract.

## 3. Core Invariants

- **OBS-1**: All observed values are **point-in-time snapshots**, not live references. Mutating the
  observed slice is undefined behavior; copies are explicit.
- **OBS-2**: Snapshot read costs O(network-size) at most, never O(history). History buffers are
  bounded ring buffers.
- **OBS-3**: Observability adds **zero overhead** to the training loop when no observer is attached
  (no allocations, no atomic writes from the loop).
- **OBS-4**: The protocol is **versioned**. Adding fields is a minor bump; removing or renaming is
  a major bump.

## 5. Detailed Design

### 5.1 Exposed Surfaces (proposed)

| Surface | Returns | Frequency |
| :--- | :--- | :--- |
| `Snapshot()` | Full network state copy (weights, activations, control state, loss history ring) | On demand (< 1 Hz typical) |
| `LossHistory(n uint)` | Last N loss values | On demand |
| `LayerActivation(idx uint)` | Slice copy of activations from last forward pass | On demand |
| `ControlState()` | Atomic read of current training state | Cheap, high-frequency safe |
| `Stats()` | Counters: iterations, elapsed time, samples seen | On demand |

### 5.2 Open Questions

- <!-- TBD: history buffer size — fixed default vs. WithLossHistorySize option -->
- <!-- TBD: should Snapshot() allow partial (e.g., only weights) for cost control? -->
- <!-- TBD: schema version negotiation between library and external GUI repo -->

## 6. Implementation Notes

1. Layer 2 strategies: (a) `l2-logging-strategy.md` — periodic snapshot to slog. (b) `l2-visualization-api.md`
   — HTTP endpoint serving snapshots in JSON.
2. Snapshot copy cost is acceptable because it is bounded by network size and triggered on demand.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[REF-RU]` | `.references/rustunumic/lib.rs` | Source `tracing::info/debug/trace/warn` integration — push pattern over pull contract |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #7 (visualization) + TODO #9 (logging) shared concept. |
