# Training Callbacks Implementation

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-training-callbacks.md

## Overview

Go realization of the training callback contract (`l1-training-callbacks.md`) in `pkg/nn/`.
Adds `CallbackRegistry[T]`, a per-event slice of typed function callbacks, integrated
directly into `pkg/nn/train.go`. The zero-overhead invariant (CB-3) is satisfied by nil
slice checks — no allocation occurs when no callbacks are registered. Panic recovery (CB-5)
wraps each individual callback invocation.

## Related Specifications

- [l1-training-callbacks.md](l1-training-callbacks.md) — Parent
- [l2-training-loop.md](l2-training-loop.md) — `train.go` integration; callback dispatch points
- [l2-nn-facade.md](l2-nn-facade.md) — `WithOnIterationEnd` / `WithOnTrainEnd` options
- [l2-errors-impl.md](l2-errors-impl.md) — `CallbackPanic` error category sentinel
- [l2-checkpointing-impl.md](l2-checkpointing-impl.md) — `OnImprovementFound` natural trigger for saves

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| CB-1 (Synchronous Invocation) | Callbacks invoked via `fireEvent(registry, ctx)` called inline within `train.go`'s iteration loop; goroutine is blocked until all registered callbacks return |
| CB-2 (No Direct Mutation) | `CallbackContext[T]` holds a `NetworkSnap` (read-only snapshot per OBS-1); no mutable NN reference is passed; network methods are not callable from callback context |
| CB-3 (Zero Overhead) | Each event slot is `[]CallbackFn[T]`; nil slice short-circuits `fireEvent` before any allocation or iteration; verified by `BenchmarkNoCallbacks` (0 allocs/op) |
| CB-4 (Read-Only State) | `CallbackContext[T]` fields are all value types or immutable snapshot slices; no pointer to internal training state is exposed |
| CB-5 (Panic Recovery) | `invokeOne(fn, ctx)` wraps each call in `recover()`; on panic emits `pkg/utils.Errorf(CallbackPanic, ...)` log; training continues |
| CB-6 (StopTraining Signal) | `CallbackFn[T]` returns `error`; callers check `errors.Is(err, ErrStopTraining)`; on match, training enters min-loss rollback path identical to loss-limit exit in `l2-training-loop` |
| CB-7 (Registration Order) | `fireEvent` iterates slice in order; first `ErrStopTraining` returns immediately (remaining callbacks in that event slot are not called) |
| CB-8 (OnTrainEnd Guarantee) | `defer fireOnTrainEnd(registry, &result)` placed at the top of `Train()`; fires regardless of return path (normal, error, panic recovery) |
| CB-9 (Event Boundary Atomicity) | `OnIterationEnd` / `OnImprovementFound` dispatched after the weight update block (`opt.Step` + `reg.ApplyMask`) has returned; no partial state is observable |

## 5. Detailed Design

### 5.1 File Structure

```plaintext
pkg/nn/
├── callbacks.go       // CallbackRegistry[T], CallbackContext[T], CallbackFn[T],
│                      //   ErrStopTraining sentinel, fireEvent, invokeOne (panic recovery)
├── callbacks_test.go  // unit: registration order, StopTraining propagation,
│                      //   panic recovery, OnTrainEnd guarantee, BenchmarkNoCallbacks
└── train.go           // modified: defer fireOnTrainEnd; fireEvent calls at iteration boundaries
```

### 5.2 Types Reference [REFERENCE]

```go
// ErrStopTraining is the sentinel a callback returns to request early termination.
var ErrStopTraining = errors.New("stop training")

// StopReason classifies why training ended.
type StopReason int

const (
    StopLossLimit      StopReason = iota
    StopMaxIterations
    StopContextCancel
    StopExternalStop
    StopCallback       // returned by a callback returning ErrStopTraining
    StopLoopError
)

// CallbackContext carries the read-only event payload passed to every callback.
type CallbackContext[T utils.Float] struct {
    Iteration int
    Loss      T
    MinLoss   T
    MinIter   int
    StopReason *StopReason // non-nil for OnTrainEnd only
    Snapshot  *Snapshot[T] // read-only; nil for OnTrainEnd if network is Stopped
}

// CallbackFn is the function type all callbacks must satisfy.
type CallbackFn[T utils.Float] func(ctx CallbackContext[T]) error

// CallbackRegistry holds per-event callback slices.
type CallbackRegistry[T utils.Float] struct {
    OnIterationEnd    []CallbackFn[T]
    OnImprovementFound []CallbackFn[T]
    OnTrainEnd        []CallbackFn[T]
}
```

### 5.3 Option Wiring in pkg/nn [REFERENCE]

```go
func WithOnIterationEnd[T utils.Float](fn CallbackFn[T]) Option[T]
func WithOnImprovementFound[T utils.Float](fn CallbackFn[T]) Option[T]
func WithOnTrainEnd[T utils.Float](fn CallbackFn[T]) Option[T]
```

Multiple calls append to the respective slice (registration order preserved per CB-7).

### 5.4 train.go Integration Sketch

```text
func (nn *NN[T]) Train(ctx, input, target):
    defer fireOnTrainEnd(nn.callbacks, &stopReason, &result)
    ...
    for iter := 1; iter <= maxIter; iter++:
        ...forward... ...loss... ...backward... ...weight update...
        if loss < minLoss:
            minLoss = loss; minIter = iter; snapshot = copy(weights)
            if stop := fireEvent(nn.callbacks.OnImprovementFound, ctx); stop: goto done
        if stop := fireEvent(nn.callbacks.OnIterationEnd, ctx); stop: goto done
    done:
        weights = snapshot  // rollback per TRN-3
        return result
```

## 6. Implementation Notes

1. `ErrStopTraining` is a package-level sentinel in `callbacks.go`; wrap with `%w` when
   passing upward so `errors.Is` still works.
2. `fireOnTrainEnd` uses `defer` — capture `stopReason` by pointer so it reflects the final
   state at defer fire time.
3. `invokeOne` must not call `log.Fatal` / `os.Exit` — only `recover()` + structured log.
4. `BenchmarkNoCallbacks` must pass `0 allocs/op` to satisfy CB-3; add to CI gate.
5. `pkg/utils/errors.go` gains new sentinel `ErrCallbackPanic` in the `CallbackPanic` category.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[L1]` | `.design/main/specifications/l1-training-callbacks.md` | Parent — CB-1..9 event contract |
| `[TRAIN]` | `pkg/nn/train.go` | Integration target — iteration dispatch points |
| `[ERRORS]` | `pkg/utils/errors.go` | `ErrCallbackPanic` sentinel addition |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-11 | Initial Stable — pkg/nn/callbacks.go blueprint; CB-1..9 compliance; CallbackRegistry[T]/CallbackFn[T] types; option wiring; train.go integration sketch. C9 Trust Mode. |
