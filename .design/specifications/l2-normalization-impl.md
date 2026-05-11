# Normalization Layer Implementation

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-normalization-layers.md

## Overview

Go realization of the normalization layer contract (`l1-normalization-layers.md`) in
`pkg/layer/norm/`. Provides three concrete types — `BatchNorm[T]`, `LayerNorm[T]`, and
`GroupNorm[T]` — all implementing a common `Normalizer[T]` interface that composites with
the existing `Layer[T]` hierarchy. Affine parameters (γ, β) participate in the optimizer
cycle via the same gradient-slot mechanism as Dense layer weights.

## Related Specifications

- [l1-normalization-layers.md](l1-normalization-layers.md) — Parent
- [l2-layer-types.md](l2-layer-types.md) — Existing layer hierarchy; `Normalizer[T]` extends it
- [l2-nn-facade.md](l2-nn-facade.md) — `WithNorm` / `WithBatchNorm` / `WithLayerNorm` options wired here
- [l2-training-loop.md](l2-training-loop.md) — `SetTrain()`/`SetEval()` propagated at loop boundaries
- [l2-init-impl.md](l2-init-impl.md) — Affine init: γ = 1, β = 0 at construction
- [l2-persistence-impl.md](l2-persistence-impl.md) — Running stats + affine params round-trip via JSON

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| NORM-1 (Shape Preservation) | `Forward(x []T) []T` — output slice is same length as input; verified by `TestShapePreservation` table-driven test across all three types |
| NORM-2 (Axis Contract) | BatchNorm computes `batchMean/batchVar` over a provided batch slice; LayerNorm over `x` directly; GroupNorm splits `x` into G groups — all in `forward_impl` internal helpers |
| NORM-3 (Affine Transform) | `gamma []T` + `beta []T` fields on each type; `applyAffine(xHat []T)` shared helper; disabled if `affine == false` (identity path) |
| NORM-4 (Epsilon) | `eps T` field, default `1e-5`; passed to `stddev(variance, eps)` helper; immutable after `New*[T]` construction |
| NORM-5 (Running Stats) | `BatchNorm[T]` only: `runningMean, runningVar []T` pre-allocated at construction; updated via `updateRunning(batchMean, batchVar T)` EMA helper; default momentum 0.1 |
| NORM-6 (Train/Eval Mode) | `mode atomic.Int32` on each type (0=Train, 1=Eval); `SetMode(m NormMode)` method; `Forward` branches on mode for BatchNorm stat source |
| NORM-7 (Weight Participation) | `GradSlots() (gamma, beta []T)` method returns gradient accumulation slices; `pkg/nn/train.go` calls these alongside layer weight slots |
| NORM-8 (Topology Integration) | `Normalizer[T]` satisfies the same `Layer[T]` interface methods (`InputSize/OutputSize/Forward`); insertable via `WithNormAfterLayer(idx int, norm Normalizer[T])` option |
| NORM-9 (Persistence Round-Trip) | `MarshalJSON`/`UnmarshalJSON` on each type encodes `running_mean`, `running_var`, `gamma`, `beta`, `eps`, `momentum`, `mode`; tested by `TestJSONRoundTrip` |

## 5. Detailed Design

### 5.1 Package Structure

```plaintext
pkg/layer/norm/
├── norm.go           // Normalizer[T] interface + NormMode enum + helpers (stddev, applyAffine)
├── batchnorm.go      // BatchNorm[T]: running stats, EMA update, train/eval dispatch
├── layernorm.go      // LayerNorm[T]: per-sample stats, no running state
├── groupnorm.go      // GroupNorm[T]: G-group partition, per-group stats
└── norm_test.go      // table-driven tests: shape, affine on/off, mode switch,
                      //   BatchNorm B=1 guard, GroupNorm G divisibility, JSON round-trip
```

### 5.2 Interface Reference [REFERENCE]

```go
type NormMode int32

const (
    NormTrain NormMode = 0
    NormEval  NormMode = 1
)

type Normalizer[T utils.Float] interface {
    Forward(x []T) []T
    SetMode(m NormMode)
    GradSlots() (gamma, beta []T) // nil if affine disabled
    InputSize() int
    OutputSize() int // always == InputSize()
}

func NewBatchNorm[T utils.Float](features int, opts ...BatchNormOption[T]) *BatchNorm[T]
func NewLayerNorm[T utils.Float](features int, opts ...LayerNormOption[T]) *LayerNorm[T]
func NewGroupNorm[T utils.Float](features, groups int, opts ...GroupNormOption[T]) *GroupNorm[T]
```

### 5.3 Option Wiring in pkg/nn [REFERENCE]

```go
// Inserts a normalizer after layer at index idx in the hidden stack.
// Network is rebuilt at Compile() with the norm layer injected.
func WithNormAfterLayer[T utils.Float](idx int, n layer.Normalizer[T]) Option[T]
// Convenience wrappers:
func WithBatchNorm[T utils.Float](idx int) Option[T]  // default BatchNorm[T](hidden[idx].Size())
func WithLayerNorm[T utils.Float](idx int) Option[T]
```

### 5.4 Train/Eval Propagation

`pkg/nn/nn.go` `SetTrain()` / `SetEval()` methods iterate the layer stack and call
`norm.SetMode(NormTrain)` / `norm.SetMode(NormEval)` atomically on all `Normalizer[T]`
instances found in the stack.

### 5.5 BatchNorm B=1 Guard

`BatchNorm.Forward` in `NormTrain` mode returns `ErrBatchNormSingleSample` if called with
a single-sample batch (detected via zero-variance shortcut). Caller is expected to switch
to `LayerNorm` for single-sample inference.

## 6. Implementation Notes

1. `norm.go` — `Normalizer[T]` interface + shared `stddev`, `applyAffine` free functions. No state.
2. `batchnorm.go` — most complex; start here. Running stats pre-allocated in `NewBatchNorm`.
3. `layernorm.go` and `groupnorm.go` — stateless variants; implement after BatchNorm passes tests.
4. `pkg/nn/train.go` — add `propagateMode(NormTrain/NormEval)` call at training loop entry/exit.
5. `pkg/nn/options.go` — add `WithNormAfterLayer`, `WithBatchNorm`, `WithLayerNorm` options.
6. `pkg/nn/compile.go` — wire norm layer gradient slots into the optimizer update pass.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[L1]` | `.design/specifications/l1-normalization-layers.md` | Parent contract — NORM-1..9 invariants |
| `[LAYER-H]` | `pkg/layer/` | Existing layer interface to extend |
| `[OPT]` | `pkg/optimizer/optimizer.go` | Gradient slot pattern for affine params |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-11 | Initial Stable — pkg/layer/norm/ blueprint; NORM-1..9 compliance table; Normalizer[T] interface; option wiring. C9 Trust Mode auto-promoted. |
