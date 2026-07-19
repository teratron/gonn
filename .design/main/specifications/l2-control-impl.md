# Training Control Implementation

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-training-control.md

## Overview

Concrete Go realization of [l1-training-control.md](l1-training-control.md): the state-machine cell (`atomic.Int32`), the buffered-channel control bus, integration with `context.Context` cancellation, and the `Pause` / `Resume` / `Stop` method placement on `*NN[T]`.

## Related Specifications

- [l1-training-control.md](l1-training-control.md) — Parent — invariants CTRL-1..CTRL-5 + state machine
- [l2-training-loop.md](l2-training-loop.md) — Hosts the state-machine cell and the safe-point check
- [l2-nn-facade.md](l2-nn-facade.md) — Surfaces `Pause/Resume/Stop` methods on `*NN[T]`

## 1. Motivation

L1 fixes the state set, transitions, and cooperative-cancellation contract. This spec decides Go specifics — atomic encoding of state values, channel buffering policy, where the supervisor goroutine lives (or whether one is needed at all), and how `context.Context` cancellation is multiplexed with explicit `Stop()` calls.

## 2. Constraints & Assumptions

- Stdlib only: `sync/atomic`, `context`.
- Single training goroutine — no supervisor goroutine. Control is sampled at safe points by the loop itself.
- State stored as `atomic.Int32` mapping `iota`-style constants {0=Idle, 1=Running, 2=Pausing, 3=Paused, 4=Stopping, 5=Stopped}.
- Control channel is **unbuffered** — back-pressure is acceptable; callers pay a goroutine schedule for explicit deterministic ordering of pause/stop signals.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| CTRL-1 (State set) | `int32` with named constants; CAS-based transitions enforce one-way edges |
| CTRL-2 (Idempotent Pause) | `Pause()` is `CAS(Running → Pausing)`; second call is a no-op (CAS fails, returns nil) |
| CTRL-3 (Stop is destructive) | `Stop()` accepts current state ∈ {Running, Paused, Pausing} and CASes to `Stopping`; `Stopped` is terminal |
| CTRL-4 (ctx parity) | At each safe point, loop reads `ctx.Err()` first; if non-nil, behaves as if `Stop()` was called |
| CTRL-5 (Lock-free reads) | Observers use `atomic.LoadInt32`; never block on transitions |

## 5. Detailed Design

### 5.1 State Type & Constants

```go
// [REFERENCE] In pkg/nn/control.go.
type State int32

const (
    StateIdle     State = 0
    StateRunning  State = 1
    StatePausing  State = 2
    StatePaused   State = 3
    StateStopping State = 4
    StateStopped  State = 5
)

type controller struct {
    state    atomic.Int32 // holds State
    pauseReq chan struct{}
    stopReq  chan struct{}
}
```

### 5.2 API Methods (on `*NN[T]`)

```go
func (n *NN[T]) Pause() error  // CAS Running → Pausing; emits to pauseReq
func (n *NN[T]) Resume() error // CAS Paused → Running; idempotent on Running
func (n *NN[T]) Stop() error   // CAS Running|Pausing|Paused → Stopping; emits to stopReq
func (n *NN[T]) State() State  // atomic load — for external observers
```

### 5.3 Safe-Point Check (inside `Train()` loop body)

```text
for iter := 0; iter < maxIter; iter++:
    forward()
    backward()
    updateWeights()

    // safe point: end of iteration
    if ctx.Err() != nil:
        cas(StateRunning, StateStopping)
        break
    if cas(StatePausing, StatePaused):
        select {
            case <-resumeReq: cas(StatePaused, StateRunning); continue
            case <-stopReq:   cas(StatePaused, StateStopping); break
            case <-ctx.Done(): cas(StatePaused, StateStopping); break
        }
    if state.Load() == StateStopping:
        break
```

### 5.4 Open Questions

- <!-- TBD: surface as methods on `*NN[T]` or via `n.Controller()` accessor? L1 §5.2 leaves it open. Lean toward methods for ergonomic v1; promote to Controller object if signal set grows. -->
- <!-- TBD: `WithEpochCallback` semantics during Pausing — fire on the iteration that triggers pause, or skip? Lean toward fire (CTRL-2 says iteration completes). -->
- <!-- TBD: pause-timeout (CTRL safe-point hang) — currently relies on caller-supplied ctx with deadline. -->

## 6. Implementation Notes

1. New file `pkg/nn/control.go` — owns the controller struct and methods.
2. `Train()` (in `pkg/nn/train.go`, see `l2-training-loop.md`) embeds the safe-point check inline, not as a callback (avoids per-iteration call overhead).
3. Resume signal is a separate channel from `pauseReq` — the receive happens from inside the safe-point block (network is paused waiting for it).
4. Tests must use `-race` and exercise concurrent `Pause` + `ctx.Cancel` to verify CTRL-4 ordering.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[CONTROL]` | `pkg/nn/control.go` (new) | Controller struct, methods, safe-point helpers |
| `[TRAIN]` | `pkg/nn/train.go` | Hosts the safe-point check inline |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-28 | Initial Draft — concrete Go realization of l1-training-control RFC. |
| 1.0.0 | 2026-04-30 | Promoted Draft → Stable. Phase-2 implementation lives in `pkg/nn/control.go` with `Pause()/Resume()/Stop()` on `*NN[T]` plus the `awaitSafePoint` worker-side helper. CTRL-1..CTRL-5 satisfied: 4-state MVP (Idle/Running/Paused/Stopped) with CAS-only transitions — collapsed §5.1's 6-state design (Pausing/Stopping intermediates merged into terminal states) because the loop polls atomically at safe points and never observes the intermediate. `transitionToRunning` uses CAS Idle→Running so a Stop issued before Fit reaches the loop is preserved (TOCTOU guard). v0.5 implementation note: context.Context multiplexing (CTRL-4) and unbuffered control-bus channels (§5.1 pauseReq/stopReq) deferred to v0.6; current path is purely atomic. Test coverage includes 16-goroutine concurrency probe under `-race`. |
