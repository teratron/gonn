# Convolutional Layers — Go Implementation

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-conv-layers.md

## Overview

Go realization of the 1-D convolutional layer contract from `l1-conv-layers.md`.
Introduces `pkg/layer/conv/` as a new sub-package with three types: `Conv1D[T]`,
`MaxPool1D[T]`, and `Flatten[T]`. All three implement the same `layer.Layer[T]`
interface used by `Input`, `Dense`, and `Output` so the existing network graph
treats them uniformly. New functional options `WithConv1D` and `WithPooling` are
added to `pkg/nn/options.go`.

## Related Specifications

- [l1-conv-layers.md](l1-conv-layers.md) — L1 parent (Draft v0.1.0)
- [l2-layer-types.md](l2-layer-types.md) — Layer[T] interface; Conv1D must satisfy it
- [l2-network-graph.md](l2-network-graph.md) — network graph iterates layers uniformly
- [l2-nn-facade.md](l2-nn-facade.md) — WithConv1D/WithPooling are new options
- [l2-init-impl.md](l2-init-impl.md) — kernel weight initialization
- [l2-optimizer-impl.md](l2-optimizer-impl.md) — optimizer Step() must reach conv weights
- [l2-perf-impl.md](l2-perf-impl.md) — sync.Pool for gradient scratch buffers
- [l2-errors-impl.md](l2-errors-impl.md) — new ErrConvShapeMismatch sentinel

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| CONV-1 Output shape formula | `outputLen()` pure function on Conv1D fields; called in `compile()` to size next layer |
| CONV-2 Filter weights only trainable params | Only `weights` (and `biases`) participate in optimizer Step(); Pool/Flatten have no params |
| CONV-3 Forward determinism | Pure function of inputs and weights; no RNG in forward path |
| CONV-4 Gradient correctness (cross-correlation) | `CalculateError` computes ∂L/∂W and ∂L/∂X via explicit loops |
| CONV-5 Pool semantics (max/avg, non-overlapping) | `MaxPool1D.CalculateValue` argmax tracking for backward pass |
| CONV-6 Flatten concatenation | `Flatten.CalculateValue` reshapes feature map to 1-D vector |
| CONV-7 Weight persistence (JSON round-trip) | `MarshalJSON/UnmarshalJSON` on Conv1D; shape metadata stored alongside weights |
| CONV-8 Weight initialization | `Init()` delegates to `utils.InitWeights` (Xavier/He/Random) |
| CONV-9 Layer interface compatibility | All three types implement `layer.Layer[T]`; compile-time assertion in `conv.go` |

## 5. Detailed Design

### 5.1 Package Structure

```plaintext
pkg/layer/conv/
├── conv.go        # package doc + compile-time interface assertions
├── conv1d.go      # Conv1D[T]: constructor, Forward, Backward, Init, JSON
├── pool.go        # MaxPool1D[T], AvgPool1D[T]: constructor, Forward, Backward
└── flatten.go     # Flatten[T]: stateless shape-collapse
```

### 5.2 Conv1D Weight Storage

<!-- TODO: User contribution — choose and document the weight storage strategy.
     See conversation for context on Variant A (flattened []T) vs Variant B ([][]T per-filter).
     Write the Conv1D[T] struct fields + a brief comment justifying the choice (5-10 lines). -->

```
// [REFERENCE] — fill in the struct fields below
type Conv1D[T utils.Float] struct {
    numFilters int
    kernelSize int
    stride     int
    padding    PadMode

    // TODO: choose weight storage format here
    // weights ???

    biases     []T     // nil if bias disabled (CONV-C7)
    gradW       ???    // gradient buffer reused across iterations (sync.Pool pattern)
    gradX       []T    // input gradient buffer

    // saved for backward pass
    lastInput   []T
    lastOutput  []T
}
```

### 5.3 MaxPool1D Backward (argmax tracking)

MaxPool requires knowing which position was the maximum during forward in order to
route the upstream gradient only to that position during backward.

```
// [REFERENCE]
type MaxPool1D[T utils.Float] struct {
    poolSize int
    argmax   []int   // position of max per window, saved during Forward
}
// Backward: upstream[i] flows only to lastInput[argmax[i]]
```

### 5.4 Flatten

Flatten is stateless — it carries no weights and its backward pass is a reshape-only
operation (fan the incoming gradient back into the feature-map shape).

### 5.5 Options Wiring

```
// [REFERENCE] — new entries in pkg/nn/options.go
func WithConv1D[T](numFilters, kernelSize, stride int, pad PadMode) Option[T]
func WithMaxPool1D[T](poolSize int) Option[T]
func WithFlatten[T]() Option[T]
```

These options append to a `[]layer.Layer[T]` convolution prefix in `config.go`.
`compile()` prepends this slice before the Dense hidden stack.

### 5.6 compile() integration

```mermaid
graph LR
    Opts[WithConv1D + WithPooling + WithFlatten] --> Compile[compile]
    Compile --> ConvStack[conv layers prepended]
    ConvStack --> DenseStack[existing hidden Dense layers]
    DenseStack --> Out[Output layer]
```

### 5.7 Error Handling

- `ErrConvShapeMismatch` — input length < kernel size (violates CONV-1 PadValid constraint).
- `ErrConvPoolSizeMismatch` — pool size > feature map length.

## 6. Implementation Notes

1. Write `pkg/layer/conv/conv.go` first (interface assertion only) — forces compiler errors early.
2. `conv1d.go` — implement `Forward` before `Backward`; use table-driven tests to verify CONV-1.
3. `pool.go` — `MaxPool1D` needs `argmax` slice allocated once in constructor (same size as output).
4. Coordinate with `pkg/nn/compile.go`: `outputLen()` must be called at compile time to set the input size of the first Dense layer.
5. `pkg/utils/errors.go` — add `ErrConvShapeMismatch`, `ErrConvPoolSizeMismatch`.

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-12 | Initial Draft — Conv1D/MaxPool1D/Flatten; weight storage TODO left for user contribution. |
| 0.1.0 | 2026-05-14 | Promoted Draft → Stable via magic.task Trust Mode (parent `l1-conv-layers` Stable; MVC satisfied). |
