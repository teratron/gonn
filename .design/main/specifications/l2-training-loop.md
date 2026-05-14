# Training Loop Implementation

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-training-semantics.md

## Overview

Concrete realization of [l1-training-semantics.md](l1-training-semantics.md) in Go: the `Train()`
function body, snapshot mechanics for min-loss rollback (TRN-3), iteration-boundary safe-points
shared with `l1-training-control.md`, and the data path between forward/backward and the network
graph.

## Related Specifications

- [l1-training-semantics.md](l1-training-semantics.md) — Parent — invariants TRN-1..TRN-6
- [l1-training-control.md](l1-training-control.md) — Pause/Resume/Stop sit on top of this loop
- [l1-checkpointing.md](l1-checkpointing.md) — Periodic snapshot triggers attach here
- [l2-network-graph.md](l2-network-graph.md) — Forward/Backward methods this loop drives
- [l2-nn-facade.md](l2-nn-facade.md) — Public `Train()` entry that delegates here

## 1. Motivation

`l1-training-semantics.md` defines what `Train()` must do; this spec defines **how** it does it in Go
without losing the invariants. Captures concrete decisions deferred at L1: type of the snapshot
buffer, ownership of weights during rollback, control-state polling cadence, and callback dispatch.

## 2. Constraints & Assumptions

- File: `pkg/nn/train.go` — currently a stub (per `l2-nn-facade.md` v1.0 known issues).
- Single goroutine per `*NN[T]` instance during training. Concurrent training across separate
  networks is fine; this spec does not address it.
- Snapshot = full weights deep-copy. Acceptable cost because allocation is amortized over `min_iter`
  intervals (typically dozens to hundreds).

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| TRN-1 (Termination) | Loop checks `(loss < lossLimit) \|\| (iter >= max) \|\| ctx.Done() \|\| stopFlag.Load()` at iteration boundary. |
| TRN-2 (min-loss tracking) | Local vars `minLoss`, `minIter` updated per iteration; `bestSnapshot` deep-copied on improvement. |
| TRN-3 (Rollback) | On termination via TRN-1 (a)/(b), `network.LoadWeights(bestSnapshot)` restores best state. |
| TRN-4 (Diagnostic return) | Returns `(uint(minIter), T(minLoss))`, never the wall-clock final iter/loss. |
| TRN-5 (Atomic iteration) | Forward → loss → backward → update is one straight-line block; no Pause/Stop check inside. |
| TRN-6 (Determinism) | Inputs the loop touches are owned by the network (weights, RNG); deterministic given config. |

## 5. Detailed Design

### 5.1 Function Skeleton (pseudo-code)

```text
func (n *NN[T]) Train(ctx Context, input, target []T) (uint, T) {
    if !n.compiled: return 0, 0  // ErrUnsupported per l1-error-taxonomy
    n.lazyInit(input, target)

    var (
        minLoss      T = MaxValue[T]()
        minIter      uint = 0
        bestSnapshot WeightsBuffer[T] = nil
        loss         T
    )

    for iter := uint(1); iter <= n.maxIterations; iter++ {
        if ctx.Err() != nil || n.controlState() == Stopping { break }
        if n.controlState() == Pausing {
            n.transitionTo(Paused)
            n.waitForResumeOrStop(ctx)
            if n.controlState() == Stopping { break }
        }

        n.network.Forward(input)
        loss = n.network.ComputeLoss(target)

        if isNaN(loss) {
            n.logger.Error("training_failed", "reason", "nan_loss", "iter", iter)
            return 0, loss   // wraps ErrTrainingFailure at facade
        }

        if loss < minLoss {
            minLoss, minIter = loss, iter
            bestSnapshot = n.network.SnapshotWeights()
            if loss < n.lossLimit { break }
        }

        n.network.Backward()
        n.network.UpdateWeights(n.learningRate)
        n.fireEpochCallback(iter, loss)
        n.maybeFireSnapshot(iter)  // see l1-checkpointing TRN integration
    }

    if minIter > 0 { n.network.LoadWeights(bestSnapshot) }
    return minIter, minLoss
}
```

### 5.2 Snapshot Buffer Type

```go
// [REFERENCE] Internal type — not exported in v0.5.
type WeightsBuffer[T utils.Float] struct {
    layers [][][]T  // [layerIdx][neuronIdx][axonIdx] = weight
    biases [][]T    // [layerIdx][neuronIdx] = bias
}
```

Owned by the loop, not by the network. Discarded when `Train()` returns.

### 5.3 Open Questions

- <!-- TBD: pool snapshot buffers via sync.Pool to reduce allocation churn across consecutive Train calls -->
- <!-- TBD: how does AndTrain (continuation) interact — fresh minLoss tracker or carry forward? -->
- <!-- TBD: callback panic recovery — should a panicking EpochCallback abort training or be logged + continue? -->

## 6. Implementation Notes

1. Replace the current stub in `pkg/nn/train.go` with this skeleton.
2. The forward/backward/update primitives already exist in `pkg/network/propagation.go` — wire them.
3. Lazy init (creating cells/axons on first `Train()` call) reads `n.compiled` flag from the v2.0
   facade lifecycle.
4. Control-state polling uses atomic load (`l1-training-control` CTRL-5) — cheap.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[TRAIN]` | `pkg/nn/train.go` | Implementation home |
| `[NETWORK]` | `pkg/network/propagation.go` | Forward/Backward primitives the loop drives |
| `[REF-RU]` | `.references/rustunumic/train.rs` | Source of the loop pattern |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-28 | Initial Draft — concrete Go realization of l1-training-semantics RFC. |
| 1.0.0 | 2026-04-30 | Promoted Draft → Stable. Phase-2 implementation lives in `pkg/nn/train.go` as the dual `Train(input, target)` (single-step) + `Fit(dataset)` (multi-epoch) surface. TRN-1..TRN-6 satisfied: termination via `LossLimit`/`MaxIterations`/`Stop()`; min-loss tracking with `snapshotWeights`/`restoreWeights`; flat `[]T` snapshot buffer (instead of nested `WeightsBuffer` from §5.2 — flat is allocation-friendly and survives Phase-1 single-hidden topology). v0.5 implementation note: `context.Context` multiplexing (TRN-1 c) deferred to v0.6; current `awaitSafePoint` honours the package-local atomic Stop signal but not external ctx. NaN-loss detection deferred to v0.6 (TRN-3 partial). |
