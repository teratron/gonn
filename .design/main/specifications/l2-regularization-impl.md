# Regularization Implementation

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-regularization.md

## Overview

Go realization of the regularization contract (`l1-regularization.md`) in `pkg/regularizer/`.
Provides the `Regularizer[T Float]` interface plus three concrete implementations — `L2[T]`,
`L1[T]`, `Dropout[T]` — and a `Compose[T]` combinator. Integrated with `pkg/nn` via the
`WithRegularizer(Regularizer[T])` functional option. Dropout uses the same seeded
`math/rand/v2.PCG` source established in `pkg/utils/init.go`.

## Related Specifications

- [l1-regularization.md](l1-regularization.md) — Parent — regularization contract
- [l2-training-loop.md](l2-training-loop.md) — Training loop that calls Penalty and ApplyMask
- [l2-nn-facade.md](l2-nn-facade.md) — Public option WithRegularizer
- [l2-init-impl.md](l2-init-impl.md) — RNG plumbing reused for Dropout mask sampling

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| REG-1 (Penalty) | `Penalty(weights []T) T` iterates the flat weight slice from pkg/network bundles; L2 sums squares; L1 sums absolute values; penalty scaled by λ |
| REG-2 (ApplyMask) | `ApplyMask(acts []T, training bool) []T` — L1/L2 return acts unchanged; Dropout samples a bernoulli mask and applies inverted scaling in-place |
| REG-3 (Inference determinism) | `ApplyMask(_, false)` returns acts without any RNG call in all implementations; verified by `TestInferenceDeterminism` |
| REG-4 (nil default) | `pkg/nn.compile()` keeps `cfg.Regularizer` nil by default; `train.go` guards penalty/mask calls with `if reg != nil` |
| REG-5 (Inverted dropout) | Retained activations multiplied by `1/p`; zero otherwise; verified by `TestDropoutInvertedScaling` |
| REG-6 (Compose) | `Compose[T]` sums `Penalty` across all members and chains `ApplyMask` left-to-right; verified by `TestComposeAdditivity` |

## 5. Detailed Design

### 5.1 Package Structure

```plaintext
pkg/regularizer/
├── regularizer.go      // Regularizer[T] interface + nil-guard Apply helper
├── l2.go               // L2[T] — λ × Σwᵢ²
├── l1.go               // L1[T] — λ × Σ|wᵢ|
├── dropout.go          // Dropout[T] — Bernoulli mask + 1/p inverted scaling
├── compose.go          // Compose[T] — additive penalty + sequential mask
└── regularizer_test.go // unit + integration tests
```

### 5.2 Interface Reference [REFERENCE]

```go
// Regularizer adds a generalization constraint to the training loop.
type Regularizer[T utils.Float] interface {
    Penalty(weights []T) T
    ApplyMask(activations []T, training bool) []T
}

// Compose chains multiple regularizers additively.
func Compose[T utils.Float](rs ...Regularizer[T]) Regularizer[T]
```

### 5.3 Integration with pkg/nn

In `pkg/nn/train.go` the weight-update path becomes:

```go
// [REFERENCE]
effectiveLoss := rawLoss
if reg != nil {
    effectiveLoss += reg.Penalty(net.AllWeights())
}
// ... backprop with effectiveLoss ...
if reg != nil {
    hiddenActs = reg.ApplyMask(hiddenActs, true)
}
```

During `Query()`, `ApplyMask(acts, false)` is called — guaranteed no-op for all types.

### 5.4 Dropout RNG

`Dropout[T]` holds a `*rand.Rand` seeded from the network's global RNG seed
(from `pkg/utils/init.go`). Seeded deterministically when `WithSeed(s)` is set,
otherwise from `time.Now().UnixNano()`. This mirrors the axon initialization pattern.

### 5.5 Test Matrix

| Test | Scope |
| :--- | :--- |
| `TestL2Penalty` | Golden formula: λ=0.01, known weights → expected penalty |
| `TestL1Penalty` | Golden formula: λ=0.01, known weights → expected penalty |
| `TestDropoutInvertedScaling` | Expected: retained acts × (1/p); sum of mask bits ≈ N×p |
| `TestInferenceDeterminism` | `ApplyMask(acts, false)` == acts; no RNG state consumed |
| `TestComposeAdditivity` | Compose(L2,Dropout).Penalty == L2.Penalty + Dropout.Penalty |
| `TestOverfitReduction` | Compose(L2(0.01), Dropout(0.8)) reduces overfit gap on E05 Iris |

## 6. Implementation Notes

1. `l2.go` first — no RNG, validates Penalty path end-to-end.
2. `dropout.go` second — introduces the RNG dependency; use `pkg/utils/init.go` seeding.
3. `compose.go` last — validates multi-regularizer interaction via `TestComposeAdditivity`.
4. The overfit-reduction integration test (`TestOverfitReduction`) splits the Iris CSV into
   80/20 train/test; trains with and without regularization; asserts test-loss gap ≥ 10%
   smaller with regularization.

## 7. Drawbacks & Alternatives

- Applying Dropout uniformly to every hidden layer may be too aggressive for shallow two-hidden
  networks. Users can set `p` close to 1.0 as an effective no-op for specific layers, or use
  `Compose` with a layer-selective wrapper (future extension).

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[REG-DIR]` | `pkg/regularizer/` | Implementation home — new package |
| `[TRAIN-LOOP]` | `pkg/nn/train.go` | Integration point: Penalty and ApplyMask called here |
| `[INIT-IMPL]` | `pkg/utils/init.go` | RNG source reused for Dropout mask sampling |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-05-07 | Initial — Go realization of l1-regularization for Phase 6. Trust Mode Stable. |
