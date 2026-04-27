# Meta-Learning Hooks

**Version:** 0.1.0
**Status:** Draft
**Layer:** concept

## Overview

Defines the contract by which any **scalar hyperparameter** (learning rate, loss limit, momentum,
gradient-skip threshold) can be **adaptively tuned by an inner GoNN network**, recursively. The
outer network's training loop reads the parameter from a `Tunable[T]` source; the source can be a
constant, a schedule, or another `*NN[T]` instance trained on the outer loop's metrics.

> ⚠ This is the most experimental spec in the v0.x series. Marked `Draft`. Expect substantive
> redesign before RFC.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent
- [l1-training-control.md](l1-training-control.md) — Inner-network training is itself controllable
- [l1-observability-protocol.md](l1-observability-protocol.md) — Inner network sees outer metrics via observability

## 1. Motivation

Hyperparameter tuning today is manual or grid-search. The library has all the pieces (training loop,
loss tracking, observability) to **let the network tune itself recursively**:

- Replace the constant `learningRate = 0.3` with `learningRate = innerNN.Query(currentLossHistory)`.
- Inner network learns from outer training trajectories.
- Same machinery works for any scalar — gradient-skip thresholds, batch-size schedules, etc.

The point of this spec is **reserving the API door** so future research can plug in without breaking
the public surface.

## 2. Constraints & Assumptions

- Tunability is **opt-in per parameter** — defaults remain constants. `WithLearningRate(0.3)` stays
  the simple path.
- Inner networks are **plain `*NN[T]`** — no special types. They use the same `Compile()`/`Query()`
  surface.
- **Recursion depth bounded** — at most one level of inner-network tuning is permitted in v1. A
  deeper meta-meta-learning topology requires explicit opt-in and a guard against infinite regress.

## 3. Core Invariants

- **META-1**: Every adaptive parameter is read through a `Tunable[T]` interface; constant and
  schedule sources are equivalent values of this interface.
- **META-2**: Inner-network queries during training are **synchronous and bounded** — they execute
  inline within the outer training loop's control flow, with a configurable maximum execution time.
- **META-3**: Recursion guard: a `Tunable[T]` tracks its own depth; a `*NN[T]`-backed `Tunable` may
  not host another `*NN[T]`-backed `Tunable` for the same parameter (cycle prevention).
- **META-4**: Inner-network state changes are **observable** — they flow through the same observability
  protocol as the outer network. A GUI can watch both.

## 5. Detailed Design

### 5.1 Interface Sketch

```text
type Tunable[T utils.Float] interface {
    Value(ctx TuningContext[T]) T
}

// Constant — trivial implementation
func Const[T utils.Float](v T) Tunable[T]

// Schedule — predefined curve (linear decay, cosine, exponential)
func LinearDecay[T utils.Float](from, to T, overSteps uint) Tunable[T]

// Adaptive — inner network owns the parameter
func AdaptiveByNetwork[T utils.Float](inner *NN[T]) Tunable[T]
```

### 5.2 Outer-loop Integration

At each iteration the outer loop builds a `TuningContext` snapshot (current iteration, last loss,
loss-history ring) and queries each `Tunable`. Constants short-circuit; adaptive sources run their
inner forward pass.

### 5.3 Open Questions

- <!-- TBD: TuningContext schema — what does the inner network see? -->
- <!-- TBD: training the inner network — when, on what loss? -->
- <!-- TBD: serialization of adaptive parameters in snapshots — store the inner network too? -->
- <!-- TBD: cost model — is per-iteration inner-network query feasible, or only per-epoch? -->

## 6. Implementation Notes

1. Phase 0 (this spec): publish the `Tunable[T]` interface stub; existing options that take `T` get
   parallel `WithXxxTunable(Tunable[T])` versions. Constants implement `Tunable` trivially.
2. Phase 1: ship `Const` + schedule implementations only.
3. Phase 2: ship `AdaptiveByNetwork` behind `WithExperimentalAdaptive(true)` flag. Recursion guard
   active from day one.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[META]` | `pkg/nn/tunable.go` (new) | Interface and stock implementations |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #14 — most experimental of the batch. |
