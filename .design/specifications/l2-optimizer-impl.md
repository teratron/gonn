# Optimizer Implementation

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-optimizer-strategies.md

## Overview

Go realization of the optimizer contract (`l1-optimizer-strategies.md`) in `pkg/optimizer/`.
Provides the `Optimizer[T Float]` interface plus four concrete types: `SGD[T]`,
`SGDMomentum[T]`, `Adam[T]`, and `RMSProp[T]`. Integrated with `pkg/nn` via the
`WithOptimizer(Optimizer[T])` functional option; falls back to `DefaultOptimizer[T](lr)` (SGD)
when no optimizer is configured.

## Related Specifications

- [l1-optimizer-strategies.md](l1-optimizer-strategies.md) — Parent — algorithm contract
- [l2-training-loop.md](l2-training-loop.md) — Training loop that calls Step() per weight-update cycle
- [l2-nn-facade.md](l2-nn-facade.md) — Public option WithOptimizer plugged in options.go
- [l2-checkpointing-impl.md](l2-checkpointing-impl.md) — Checkpoint snapshots must include optimizer state

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| OPT-1 (Step in-place) | `Step(weights, deltas []T) error` — weights updated in-place over the pkg/network bundle slices; deltas consumed read-only |
| OPT-2 (once per cycle) | `pkg/nn/train.go` calls `opt.Step` once per iteration inside the weight-update hot path; no re-entry guard needed — caller contract enforced by design |
| OPT-3 (zero LR = no-op) | Each implementation returns `nil` immediately when `lr == 0` without touching weights |
| OPT-4 (Reset) | `Reset()` zeroes all moment slice contents and resets step counter `t` to 0; verified by `TestResetIdempotent` |
| OPT-5 (SGD baseline) | `SGD[T]` is the package default; `DefaultOptimizer[T](lr T) Optimizer[T]` returns `NewSGD(lr)` |
| OPT-6 (serialization) | `SaveState() ([]byte, error)` / `LoadState([]byte) error` encode moment slices and step counter as JSON; verified by `TestAdamRoundTrip` |

## 5. Detailed Design

### 5.1 Package Structure

```plaintext
pkg/optimizer/
├── optimizer.go      // Optimizer[T] interface + DefaultOptimizer[T]()
├── sgd.go            // SGD[T] — stateless; O(1) memory
├── sgd_momentum.go   // SGDMomentum[T] — velocity slice; O(n) memory
├── adam.go           // Adam[T] — first + second moment slices; O(2n) memory
├── rmsprop.go        // RMSProp[T] — squared EMA slice; O(n) memory
└── optimizer_test.go // unit + benchmark tests
```

### 5.2 Interface Reference [REFERENCE]

```go
// Optimizer updates weight parameters in-place given their deltas.
type Optimizer[T utils.Float] interface {
    Step(weights, deltas []T) error
    Reset()
    LearningRate() T
    SaveState() ([]byte, error)
    LoadState([]byte) error
}

// DefaultOptimizer returns a vanilla SGD optimizer with the given learning rate.
func DefaultOptimizer[T utils.Float](lr T) Optimizer[T]
```

### 5.3 Integration with pkg/nn

`compile()` in `pkg/nn/compile.go` resolves `cfg.Optimizer`; if nil, falls back to
`optimizer.DefaultOptimizer[T](cfg.LearningRate)`. The resolved instance is stored on
`NN[T]` and passed into `pkg/nn/train.go`'s iteration loop, replacing the current inline
`rate × delta` product with `opt.Step(weights, deltas)`.

### 5.4 Test Matrix

| Test | Scope |
| :--- | :--- |
| `TestSGDGolden` | SGD golden values (lr=0.1, known delta → expected weight) |
| `TestZeroLRNoOp` | All types: zero lr leaves weights unchanged |
| `TestAdamRoundTrip` | `LoadState(SaveState())` → identical next Step result |
| `TestResetIdempotent` | Reset after N steps → same output as fresh instance |
| `BenchmarkSGDStep` | Target: 0 allocs/op for hot path |
| `BenchmarkAdamStep` | Target: 0 allocs/op (moment slices pre-allocated) |

## 6. Implementation Notes

1. `sgd.go` first — confirms the interface builds and wires into `train.go`.
2. `adam.go` second — validates state management and JSON serialization.
3. Moment slices in Adam/RMSProp are lazily allocated on first `Step` (preallocated to
   `len(weights)` on first call, then reused).
4. `optimizer_test.go` covers all six rows of the test matrix above.

## 7. Drawbacks & Alternatives

- Memory overhead for Adam: two extra `[]T` slices per weight array. Acceptable for GoNN's
  model sizes; revisit if targeting embedded / microcontroller deployments.
- Fused weight+optimizer update (single pass) would halve memory bandwidth but complicates
  the interface — deferred to a future performance optimization pass.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[OPT-DIR]` | `pkg/optimizer/` | Implementation home — new package |
| `[TRAIN-LOOP]` | `pkg/nn/train.go` | Integration point: Step called inside iteration |
| `[NN-COMPILE]` | `pkg/nn/compile.go` | Optimizer option resolved; default applied here |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-05-07 | Initial — Go realization of l1-optimizer-strategies for Phase 6. Trust Mode Stable. |
