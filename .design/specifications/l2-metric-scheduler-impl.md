# Metric-Driven Scheduler Implementation

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-lr-scheduling.md

## Overview

Go realization of the metric-driven schedulers from `l1-lr-scheduling.md` §5.5 and §5.2 —
`ReduceOnPlateau[T]` and `OneCycleLR[T]`. Both require a `MetricScheduler[T]` interface
extension that adds `StepWithMetric(metric T) T` alongside the base `Scheduler[T]` contract.
Implements the `pkg/optimizer/` additions deferred from Phase 8 ("require MetricScheduler
interface design decision" in `l2-lr-scheduling-impl.md` §5.3 Deferred list).

## Related Specifications

- [l1-lr-scheduling.md](l1-lr-scheduling.md) — Parent contract — ReduceOnPlateau §5.5, OneCycleLR §5.2
- [l2-lr-scheduling-impl.md](l2-lr-scheduling-impl.md) — Existing scheduler implementations (StepLR, WarmUpLR, CosineAnnealingLR, ChainScheduler, ExponentialLR)
- [l2-optimizer-impl.md](l2-optimizer-impl.md) — Optimizer types (SGD/Adam/RMSProp/Momentum) that schedulers wrap
- [l2-training-loop.md](l2-training-loop.md) — Training loop integration for metric dispatch
- [l2-nn-facade.md](l2-nn-facade.md) — `WithScheduler` option surface

## 1. Motivation

Phase 8 deferred `ReduceOnPlateau[T]` and `OneCycleLR[T]` with the note "require MetricScheduler
interface design decision". The issue: these schedulers observe a scalar metric (validation loss)
to decide whether to reduce the rate, but `Scheduler[T].Step()` takes no arguments. Extending
the existing interface is a breaking change. This spec defines a minimal **additive** extension:
`MetricScheduler[T]` embeds `Scheduler[T]` and adds one method. Existing code receiving
`Scheduler[T]` remains fully compatible — only the training loop needs a type-assertion to
detect and dispatch metric-aware schedulers.

## 2. Constraints & Assumptions

- `MetricScheduler[T]` **embeds** `Scheduler[T]` — no breakage to existing code.
- `Step() T` on both types calls `StepWithMetric(0)` as a safe no-op for non-metric callers
  (zero metric never triggers a plateau; OneCycleLR ignores the metric entirely).
- `ReduceOnPlateau` default granularity: `PerEpoch`; monitors loss (lower = better, `mode="min"`).
- `OneCycleLR` default granularity: `PerStep`; self-contained 3-phase curve.
- Defaults: `factor=0.1`, `patience=10`, `threshold=1e-4`, `minLR=1e-8` for ReduceOnPlateau.
- Defaults: `pctStart=0.3` (30 % warm-up), `finalDiv=1e4` for OneCycleLR.
- C29 stdlib only. C30 ≥80 % coverage. C25 generic over `utils.Float`.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| LRS-1 (Step advances) | `Step() T` delegates to `StepWithMetric(0)` — advances state correctly; metric=0 is treated as "improvement" when no prior best exists (safe no-op). |
| LRS-2 (No rate change before first Step) | Both types hold `current = lr0` at construction; constructor never calls `SetLearningRate`. |
| LRS-3 (Composable) | Both satisfy `Scheduler[T]` → valid as `ChainScheduler[T]` segments. |
| LRS-4 (Expired holds last rate) | ReduceOnPlateau: holds last reduced rate; OneCycleLR: holds `finalLR` when `step >= totalSteps`. |
| LRS-5 (Reset restores initial state) | Full state reset: patience counter, step counter, `current = lr0`, `best` reset to sentinel. |
| LRS-6 (Serialization round-trip) | Both implement `SaveState()`/`LoadState()` via JSON state structs. |

## 5. Detailed Design

### 5.1 Package Structure

```plaintext
pkg/optimizer/
├── metric_scheduler.go       # MetricScheduler[T] interface
├── reduce_on_plateau.go      # ReduceOnPlateau[T] implementation
├── reduce_on_plateau_test.go # Unit tests
├── one_cycle_lr.go           # OneCycleLR[T] implementation
└── one_cycle_lr_test.go      # Unit tests
```

### 5.2 MetricScheduler Interface

```go
// [REFERENCE]
// MetricScheduler extends Scheduler with metric-driven learning rate adjustment.
// Implementations observe a scalar training metric (e.g., validation loss) to
// decide whether to modify the learning rate.
type MetricScheduler[T utils.Float] interface {
    Scheduler[T]
    // StepWithMetric advances the schedule using the provided metric value and
    // returns the new effective learning rate. Called once per scheduling interval.
    StepWithMetric(metric T) T
}

// compile-time interface assertions in each implementation file:
var _ MetricScheduler[float32] = (*ReduceOnPlateau[float32])(nil)
var _ MetricScheduler[float32] = (*OneCycleLR[float32])(nil)
```

### 5.3 ReduceOnPlateau Algorithm

```text
StepWithMetric(metric):
    if metric improves relative to best (by >= threshold fraction, per mode):
        best = metric
        patience_count = 0
    else:
        patience_count++
    if patience_count >= patience:
        current = max(current × factor, minLR)
        patience_count = 0
        optimizer.SetLearningRate(current)
    return current
```

Improvement check:
- `mode="min"` (default — monitors loss): `metric < best × (1 − threshold)`
- `mode="max"` (monitors accuracy): `metric > best × (1 + threshold)`

Constructor: `NewReduceOnPlateau[T](opt LearningRateSetter[T], lr0 T, opts ...ReducePlateauOption[T]) *ReduceOnPlateau[T]`.

State struct for JSON round-trip:

```go
// [REFERENCE]
type reducePlateauState[T utils.Float] struct {
    LR0           T       `json:"lr0"`
    Current       T       `json:"current"`
    Best          T       `json:"best"`
    PatienceCount int     `json:"patience_count"`
    Factor        float64 `json:"factor"`
    Patience      int     `json:"patience"`
    Threshold     float64 `json:"threshold"`
    MinLR         T       `json:"min_lr"`
    Mode          string  `json:"mode"`
}
```

### 5.4 OneCycleLR Algorithm

Three-phase schedule (Smith & Touvron 2019):

```text
warmupSteps  = floor(pctStart × totalSteps)
decaySteps   = totalSteps - warmupSteps - finalSteps
finalSteps   = floor(totalSteps / (finalDiv × 2))   // small tail
minCycleLR   = maxLR / finalDiv
lr0          = maxLR / (finalDiv × divFactor)        // divFactor=25 default

Phase 1 — warm-up  (t < warmupSteps):
    lr = lr0 + (maxLR − lr0) × (t / warmupSteps)

Phase 2 — cosine decay  (warmupSteps ≤ t < warmupSteps + decaySteps):
    cycleT = t − warmupSteps
    lr = minCycleLR + 0.5 × (maxLR − minCycleLR) × (1 + cos(π × cycleT / decaySteps))

Phase 3 — final anneal  (remaining steps):
    fT = t − warmupSteps − decaySteps
    lr = minCycleLR × (1 − fT / finalSteps)
```

After `totalSteps` the rate is held at `minCycleLR × ε` per LRS-4.

Constructor: `NewOneCycleLR[T](opt LearningRateSetter[T], maxLR T, totalSteps int, opts ...OneCycleOption[T]) *OneCycleLR[T]`.

State struct:

```go
// [REFERENCE]
type oneCycleLRState[T utils.Float] struct {
    LR0        T       `json:"lr0"`
    MaxLR      T       `json:"max_lr"`
    Current    T       `json:"current"`
    Step       int     `json:"step"`
    TotalSteps int     `json:"total_steps"`
    PctStart   float64 `json:"pct_start"`
    FinalDiv   float64 `json:"final_div"`
    DivFactor  float64 `json:"div_factor"`
}
```

### 5.5 Training Loop Integration

`pkg/nn/train.go` epoch-dispatch block requires a type assertion to route metric-aware schedulers:

```go
// [REFERENCE] — addition to the per-epoch scheduler dispatch in Train()
if sched != nil && sched.Granularity() == optimizer.PerEpoch {
    if ms, ok := sched.(optimizer.MetricScheduler[T]); ok {
        ms.StepWithMetric(T(lastLoss))
    } else {
        sched.Step()
    }
}
```

`lastLoss` is the final batch loss of the epoch — already computed in the existing `Train()` loop.

## 6. Implementation Notes

1. Write `metric_scheduler.go` with the interface first — no code until interface is settled.
2. Implement `ReduceOnPlateau[T]` with functional options pattern (`WithFactor`, `WithPatience`,
   `WithMode`, `WithMinLR`, `WithThreshold`) — same pattern as other scheduler constructors.
3. Implement `OneCycleLR[T]` independently — no shared state with ReduceOnPlateau.
4. Patch `pkg/nn/train.go` epoch-dispatch block (5 lines).
5. Tests must cover: `Step()` as no-op, `StepWithMetric` plateau trigger, `mode="max"`,
   `Reset()` restores state, `SaveState`/`LoadState` round-trip, OneCycleLR 3-phase boundaries,
   `BindScheduler` wiring, `Granularity()` defaults.

## 7. Drawbacks & Alternatives

- **Alternative (callback injection)**: Training loop exposes a `LossCallback` to feed metric.
  Rejected — callbacks are not serializable and couple the user API to training internals.
- **Alternative (StepWithMetric on base Scheduler[T])**: Replace `Step() T` entirely with
  `Step(metric T) T` — simpler but breaks all existing scheduler implementations.
- **Drawback (manual dispatch for library users)**: Users who bypass `Train()` and call
  `sched.Step()` manually do not get metric injection automatically. Documented in API reference.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[SCHED]` | `pkg/optimizer/scheduler.go` | Base `Scheduler[T]` interface and `BindScheduler` — MetricScheduler embeds this |
| `[TRAIN]` | `pkg/nn/train.go` | Epoch-dispatch block receiving the type-assertion patch |
| `[STEP-LR]` | `pkg/optimizer/step_lr.go` | Pattern reference for constructor / SaveState / Reset |
| `[LR-IMPL]` | `.design/specifications/l2-lr-scheduling-impl.md` | Deferred list §5.3 that this spec closes |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-10 | Initial Stable — MetricScheduler[T] interface, ReduceOnPlateau[T], OneCycleLR[T]. Closes Phase 8 backlog. |
