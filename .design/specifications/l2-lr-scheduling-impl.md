# LR Scheduling Implementation

**Version:** 1.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-lr-scheduling.md

## Overview

Go realization of the learning rate scheduling contract (`l1-lr-scheduling.md`) in `pkg/optimizer/`.
Phase 7 shipped four scheduler types — StepLR, WarmUpLR, CosineAnnealingLR, ChainScheduler — covering
all six L1 invariants (LRS-1..LRS-6). Three optional types from the L1 taxonomy (ExponentialLR,
ReduceOnPlateau, OneCycleLR) are deferred and tracked as TODOs; their absence does not violate any
L1 invariant.

## Related Specifications

- [l1-lr-scheduling.md](l1-lr-scheduling.md) — Parent contract
- [l2-optimizer-impl.md](l2-optimizer-impl.md) — Optimizers that schedulers wrap
- [l2-training-loop.md](l2-training-loop.md) — Training loop that drives Scheduler.Step()
- [l2-deep-builder.md](l2-deep-builder.md) — Builder/Options surface that exposes WithScheduler

## 1. Motivation

`l1-lr-scheduling.md` was stabilized in Phase 7 alongside the implementation. This spec closes the
registry gap: the Go implementation in `pkg/optimizer/` is complete but had no corresponding L2 spec.

## 2. Constraints & Assumptions

- All scheduler types are generic over `utils.Float` (`float32 | float64`).
- Schedulers extend `pkg/optimizer/` — no new top-level package.
- `boundScheduler[T]` is package-private; callers use `BindScheduler[T]` exclusively.
- JSON round-trip for `SaveState`/`LoadState` uses `encoding/json` from stdlib — no extra deps.
- `ChainScheduler` delegates `Reset()` to the currently active sub-scheduler and advances via
  cumulative step counters; sub-schedulers are ordered slices, not a linked list.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| LRS-1 `Step()` returns new rate | All concrete types implement `Step() T`; rate returned and pushed to bound optimizer via `SetLearningRate`. |
| LRS-2 No mutation before first Step | `BindScheduler` reads initial rate at bind time; `boundScheduler.Step()` is the only write path to `SetLearningRate`. |
| LRS-3 `ChainScheduler` composition | `ChainScheduler[T]` in `chain_scheduler.go` — ordered `(Scheduler[T], steps uint)` pairs; transitions at cumulative step boundary. |
| LRS-4 Hold last rate after expiry | Every concrete type checks `step >= totalSteps` and returns the last computed rate without modifying it. |
| LRS-5 `Reset()` restores initial state | `Reset()` sets `step = 0` and restores `lr0` field in all concrete types. |
| LRS-6 `SaveState`/`LoadState` round-trip | All types marshal internal state (step counter, initial lr, params) to JSON; round-trip verified in `scheduler_test.go`. |

## 5. Detailed Design

### 5.1 Package Structure

```plaintext
pkg/optimizer/
├── scheduler.go          # Scheduler[T], LearningRateSetter[T], BindScheduler[T], Granularity
├── step_lr.go            # StepLR[T] — periodic step decay (lr₀ × gamma^⌊t/stepSize⌋)
├── warmup_lr.go          # WarmUpLR[T] — linear ramp-up (lr₀ × t/warmupSteps, then constant)
├── cosine_lr.go          # CosineAnnealingLR[T] — cosine decay to lr_min
├── chain_scheduler.go    # ChainScheduler[T] — ordered composition of (scheduler, steps) pairs
└── exponential_lr.go     # ExponentialLR[T] — per-step gamma decay (lr₀ × gamma^t)
```

### 5.2 Core Interfaces

```go
// [REFERENCE]
type Scheduler[T utils.Float] interface {
    Step() T
    Reset()
    Granularity() Granularity
    SaveState() ([]byte, error)
    LoadState([]byte) error
}

type LearningRateSetter[T utils.Float] interface {
    SetLearningRate(T)
}

func BindScheduler[T utils.Float](opt Optimizer[T], sched Scheduler[T]) Scheduler[T]
```

### 5.3 Scheduler Taxonomy

#### Implemented

| Type | File | Parameters | Rate formula |
| :--- | :--- | :--- | :--- |
| `StepLR[T]` | `step_lr.go` | `lr0, stepSize, gamma` | `lr0 × gamma^⌊t/stepSize⌋` |
| `WarmUpLR[T]` | `warmup_lr.go` | `lr0, warmupSteps` | `lr0 × t/warmupSteps` → `lr0` |
| `CosineAnnealingLR[T]` | `cosine_lr.go` | `lr0, T_max, lr_min` | `lr_min + 0.5(lr0−lr_min)(1+cos(πt/T_max))` |
| `ChainScheduler[T]` | `chain_scheduler.go` | `[]segment{Scheduler,steps}` | delegates to active sub-scheduler |
| `ExponentialLR[T]` | `exponential_lr.go` | `lr0, gamma` | `lr0 × gamma^t` |

#### Deferred (not yet implemented)

- `ReduceOnPlateau` — requires metric injection into `Step()` (signature change); needs `MetricScheduler[T]` interface.
- `OneCycleLR` — depends on `ReduceOnPlateau` interface design decision.

### 5.4 Granularity Dispatch in Training Loop

```go
// [REFERENCE] — pkg/nn/train.go
for epoch := range epochs {
    for _, batch := range batches {
        trainStep(...)
        if sched != nil && sched.Granularity() == PerStep {
            sched.Step()
        }
    }
    if sched != nil && sched.Granularity() == PerEpoch {
        sched.Step()
    }
}
```

`WarmUpLR` defaults to `PerStep`; all others default to `PerEpoch`. Overridable at construction.

### 5.5 Optimizer Integration

All four built-in optimizers (SGD, Adam, RMSProp, SGDMomentum) implement `LearningRateSetter[T]`
via `SetLearningRate(T)`. `BindScheduler` uses a type assertion — graceful degradation if an
optimizer does not implement the interface (scheduler advances but optimizer rate is unchanged).

## 6. Implementation Notes

1. `BindScheduler` wraps with a private `boundScheduler[T]` — callers cannot unwrap.
2. `ChainScheduler.Reset()` resets only the currently active sub-scheduler; a full reset
   rewinds to segment 0 and resets its sub-scheduler.
3. `SaveState` encodes scheduler type tag in JSON for cross-type `LoadState` validation.

## 7. Drawbacks & Alternatives

- **Alternative (callbacks)**: EpochCallback could manually adjust LR — ad-hoc, non-serializable.
  Schedulers are preferred for composability and checkpoint safety.
- **Deferred types**: ReduceOnPlateau requires metric injection into `Step()`, which changes the
  signature. A `MetricStep(metric T) T` variant is the natural extension path.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[SCHED]` | `pkg/optimizer/scheduler.go` | Core interface and BindScheduler |
| `[STEP-LR]` | `pkg/optimizer/step_lr.go` | StepLR implementation |
| `[WARMUP]` | `pkg/optimizer/warmup_lr.go` | WarmUpLR implementation |
| `[COSINE]` | `pkg/optimizer/cosine_lr.go` | CosineAnnealingLR implementation |
| `[CHAIN]` | `pkg/optimizer/chain_scheduler.go` | ChainScheduler implementation |
| `[EXP-LR]` | `pkg/optimizer/exponential_lr.go` | ExponentialLR implementation |
| `[TESTS]` | `pkg/optimizer/scheduler_test.go` | Coverage 88.2% — all types + BindScheduler + SaveState/LoadState |
| `[TRAIN]` | `pkg/nn/train.go` | Granularity dispatch integration |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-05-10 | Initial — Go realization of LR scheduling (Phase 7). Closes registry gap for pkg/optimizer/scheduler*.go. Trust Mode Stable (all LRS-1..LRS-6 invariants covered). |
| 1.1.0 | 2026-05-10 | Phase 8 Track A — ExponentialLR[T] (lr₀ × gamma^t) added; §5.1 package structure updated; §5.3 Deferred list updated. |
