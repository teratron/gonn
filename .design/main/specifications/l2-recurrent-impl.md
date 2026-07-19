# Recurrent Layers — Go Implementation

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-recurrent-layers.md

## Overview

Concrete Go realization of [l1-recurrent-layers.md](l1-recurrent-layers.md) — three recurrent
layer types `SimpleRNN[T]`, `LSTM[T]`, `GRU[T]` plus the `LastStep[T]` extractor, delivered as the
`pkg/layer/recurrent/` package. Each type satisfies the existing `layer.Layer[T]` interface so the
topology graph treats them uniformly with Dense / Conv layers. Adds five `pkg/nn/` functional
options (`WithSimpleRNN` / `WithLSTM` / `WithGRU` / `WithLastStep` / `WithGradClipNorm`), three new
error sentinels in `pkg/utils/errors.go`, and a `Step()` API on each cell type for stateful
streaming inference.

The package mirrors the layout pattern of `pkg/layer/conv/`: one file per layer type plus a shared
`cell.go` for orthogonal initialization, gate fusion helpers, and state caching utilities.

## Related Specifications

- [l1-recurrent-layers.md](l1-recurrent-layers.md) — Parent — REC-1..REC-9 invariants + REC-C1..REC-C7 constraints
- [l2-layer-types.md](l2-layer-types.md) — `Layer[T]` interface this package satisfies
- [l2-conv-layers-impl.md](l2-conv-layers-impl.md) — Reference for `pkg/layer/conv/` layout pattern and option-wiring style
- [l2-init-impl.md](l2-init-impl.md) — Weight initialization helpers (`utils.Xavier`, `utils.HeNormal`) extended with `utils.Orthogonal` for `W_hh`
- [l2-training-loop.md](l2-training-loop.md) — Backward pass dispatcher; BPTT plugs in at the layer-backward extension point
- [l2-optimizer-impl.md](l2-optimizer-impl.md) — `WithGradClipNorm` is wired into the optimizer step path
- [l2-persistence-impl.md](l2-persistence-impl.md) — JSON round-trip for recurrent weights per REC-8

## 1. Motivation

L1 fixes the math (REC-2 / REC-3 / REC-4 forward equations) and the gradient-flow contract
(REC-5 BPTT). This spec fixes the Go specifics:

- **Package path**: `pkg/layer/recurrent/` — peer to `pkg/layer/conv/`, `pkg/layer/norm/`.
- **Layer-interface conformance**: explicit `var _ layer.Layer[float64] = (*LSTM[float64])(nil)` compile-time assertions for all three types.
- **State cache layout**: `[]T` flat with size `SeqLen × Hidden` for `h`, `SeqLen × Hidden` for `c` (LSTM), `SeqLen × Hidden × 4` for fused gate activations.
- **Orthogonal init implementation**: stdlib-only QR decomposition or modified Gram-Schmidt for `W_hh`.
- **Option wiring**: five new options propagate the recurrent layer onto `config.RecurrentLayers`, consumed by `compile()` after Conv prefix and before Dense stack.
- **Streaming API**: `Step(x_t)` maintains internal state across calls; `ResetState()` clears it.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| REC-1 (Forward shape) | `Forward` accepts `(SeqLen, F_in)` flattened to `[]T` of length `SeqLen·F_in`; layer's `inputShape()` exposes `(SeqLen, F_in)` to `compile()` so shape propagation is explicit |
| REC-2 (SimpleRNN cell) | `pkg/layer/recurrent/simple_rnn.go` Forward implements `h_t = tanh(Wxh·x_t + Whh·h_{t-1} + b_h)` verbatim; tanh applied via `activation.TanH[T]` shared helper |
| REC-3 (LSTM cell) | `pkg/layer/recurrent/lstm.go` Forward implements 4 gates with fused matmul: `gates_t = sigmoid_or_tanh(W_x·x_t + W_h·h_{t-1} + b)` where `W_x`, `W_h`, `b` are concatenated `[4·Hidden, F_in]` / `[4·Hidden, Hidden]` / `[4·Hidden]`; gate selection via index slicing |
| REC-4 (GRU cell) | `pkg/layer/recurrent/gru.go` Forward implements 2 gates + candidate; same fused-matmul pattern as LSTM with `[3·Hidden, ·]` weight matrices |
| REC-5 (BPTT correctness) | `Backward` walks `t = SeqLen-1 → 0` accumulating `gradW_x`, `gradW_h`, `gradB` per step; finite-difference test in `lstm_test.go` validates within `1e-4` tolerance for `T=float64` |
| REC-6 (Initial state contract) | `h_0` (and `c_0` for LSTM) default to zero slices in `Forward`; `SetInitialState(h, c)` overrides; `ResetState()` zeroes the internal cache for `Step()` API |
| REC-7 (Weight initialization) | `Init(rng)` calls `utils.Xavier[T](rng, F_in)` for `W_x*` and `utils.Orthogonal[T](rng, Hidden)` for `W_h*`; new helper `Orthogonal` added to `pkg/utils/init.go` |
| REC-8 (Persistence) | `MarshalJSON` serialises `{Type, Hidden, Wxh, Whh, Bh}` (per cell-type schema); `UnmarshalJSON` validates length parity; initial state explicitly NOT serialised |
| REC-9 (Layer interface compatibility) | Compile-time assertions `var _ layer.Layer[float64] = (*SimpleRNN[float64])(nil)` etc. in each file; `Forward`/`Backward`/`Init`/`MarshalJSON` signatures match Conv / Dense |

## 5. Detailed Design

### 5.1 Package Layout

```text
pkg/layer/recurrent/
├── doc.go              // package documentation + AI-Meta block
├── cell.go             // shared helpers: orthogonal init wrapper, sigmoid/tanh fused, state caches
├── simple_rnn.go       // SimpleRNN[T] — tanh Elman recurrence
├── lstm.go             // LSTM[T] — 4-gate cell with fused matmul
├── gru.go              // GRU[T] — 2-gate cell + candidate
├── last_step.go        // LastStep[T] — stateless time-axis slice extractor
├── recurrent_test.go   // shared helpers test
├── simple_rnn_test.go  // SimpleRNN Forward / Backward / Init / JSON round-trip
├── lstm_test.go        // LSTM full suite including finite-diff gradient check
├── gru_test.go         // GRU full suite
└── last_step_test.go   // LastStep table-driven slicing
```

### 5.2 SimpleRNN Struct

```go
// [REFERENCE] In pkg/layer/recurrent/simple_rnn.go.
package recurrent

type SimpleRNN[T utils.Float] struct {
    SeqLen   int
    InSize   int
    Hidden   int

    Wxh []T // [Hidden, InSize] row-major
    Whh []T // [Hidden, Hidden] row-major
    Bh  []T // [Hidden]

    // BPTT cache — populated by Forward, consumed by Backward.
    lastInput  []T // [SeqLen, InSize]
    lastHidden []T // [SeqLen+1, Hidden] including h_0 at index 0

    // Gradient slots — pre-allocated; zeroed at the start of each Backward.
    gradWxh []T
    gradWhh []T
    gradBh  []T
    gradX   []T

    // Step() API state — separate from BPTT cache.
    stepH []T // [Hidden] running hidden state across Step calls
}

var _ layer.Layer[float64] = (*SimpleRNN[float64])(nil)
```

### 5.3 LSTM Struct

```go
// [REFERENCE] In pkg/layer/recurrent/lstm.go.
package recurrent

type LSTM[T utils.Float] struct {
    SeqLen   int
    InSize   int
    Hidden   int

    // Fused gate matrices: concatenated [i, f, g, o] along leading axis.
    Wx []T // [4*Hidden, InSize] row-major
    Wh []T // [4*Hidden, Hidden] row-major
    B  []T // [4*Hidden]

    // BPTT cache.
    lastInput      []T // [SeqLen, InSize]
    lastHidden     []T // [SeqLen+1, Hidden]
    lastCell       []T // [SeqLen+1, Hidden]
    lastGateActiv  []T // [SeqLen, 4*Hidden] post-activation (i, f, g, o)

    // Gradient slots.
    gradWx []T
    gradWh []T
    gradB  []T
    gradX  []T

    // Step() API state.
    stepH []T
    stepC []T
}

var _ layer.Layer[float64] = (*LSTM[float64])(nil)
```

### 5.4 GRU Struct

```go
// [REFERENCE] In pkg/layer/recurrent/gru.go.
package recurrent

type GRU[T utils.Float] struct {
    SeqLen   int
    InSize   int
    Hidden   int

    // Fused gate matrices: concatenated [r, z, h̃] along leading axis.
    Wx []T // [3*Hidden, InSize]
    Wh []T // [3*Hidden, Hidden]
    B  []T // [3*Hidden]

    // BPTT cache.
    lastInput     []T // [SeqLen, InSize]
    lastHidden    []T // [SeqLen+1, Hidden]
    lastGateActiv []T // [SeqLen, 3*Hidden] (r, z, h̃)

    // Gradient slots.
    gradWx []T
    gradWh []T
    gradB  []T
    gradX  []T

    // Step() API state.
    stepH []T
}

var _ layer.Layer[float64] = (*GRU[float64])(nil)
```

### 5.5 LastStep Struct (stateless)

```go
// [REFERENCE] In pkg/layer/recurrent/last_step.go.
package recurrent

// LastStep extracts the final time step of a recurrent output.
// Forward: input shape (SeqLen, Hidden) → output shape (Hidden,).
// Backward: route upstream gradient to time index SeqLen-1; zero elsewhere.
type LastStep[T utils.Float] struct {
    SeqLen int
    Hidden int
}

var _ layer.Layer[float64] = (*LastStep[float64])(nil)
```

### 5.6 Functional Options (`pkg/nn/options.go`)

```go
// [REFERENCE] Additions to pkg/nn/options.go.

func WithSimpleRNN[T utils.Float](seqLen, inSize, hidden int) Option[T] {
    return func(c *config[T]) {
        c.RecurrentLayers = append(c.RecurrentLayers, recurrent.NewSimpleRNN[T](seqLen, inSize, hidden))
    }
}

func WithLSTM[T utils.Float](seqLen, inSize, hidden int) Option[T] {
    return func(c *config[T]) {
        c.RecurrentLayers = append(c.RecurrentLayers, recurrent.NewLSTM[T](seqLen, inSize, hidden))
    }
}

func WithGRU[T utils.Float](seqLen, inSize, hidden int) Option[T] {
    return func(c *config[T]) {
        c.RecurrentLayers = append(c.RecurrentLayers, recurrent.NewGRU[T](seqLen, inSize, hidden))
    }
}

func WithLastStep[T utils.Float]() Option[T] {
    return func(c *config[T]) {
        c.RecurrentTail = recurrent.NewLastStep[T]() // singleton tail extractor
    }
}

func WithGradClipNorm[T utils.Float](threshold float64) Option[T] {
    return func(c *config[T]) { c.GradClipNorm = threshold }
}
```

`config[T]` gains two fields: `RecurrentLayers []layer.Layer[T]` (parallel to `Conv2DPrefix`)
and `RecurrentTail layer.Layer[T]` (single optional extractor). `GradClipNorm` is a `float64`
threshold; 0 means disabled (default).

### 5.7 Compile Wiring (`pkg/nn/compile.go`)

The compile order becomes:

```text
[Input] → [Conv2D prefix] → [Conv1D prefix] → [Recurrent layers] → [Recurrent tail] → [Dense stack] → [Output]
```

`setupRecurrentShapes()` runs after the Conv prefix shape walk and before the Dense walk: it
threads the `(SeqLen, F_in)` through each recurrent layer, producing the input shape for the
tail extractor (or the next recurrent layer when stacked).

### 5.8 Orthogonal Init (`pkg/utils/init.go`)

```go
// [REFERENCE] Addition to pkg/utils/init.go.

// Orthogonal generates an N×N orthonormal matrix via modified Gram-Schmidt
// on an N×N Gaussian matrix. Returns the flattened row-major []T of length N*N.
func Orthogonal[T Float](rng *rand.Rand, n int) []T {
    // 1. A = N×N matrix of N(0, 1) samples.
    // 2. Gram-Schmidt: for each column j, subtract projections onto columns 0..j-1, then normalise.
    // 3. Return flattened row-major.
}
```

Test asserts `‖Q · Q^T - I‖_F < 1e-10` for `T=float64` and `< 1e-5` for `T=float32`.

### 5.9 Gradient Clipping (`pkg/optimizer/clip.go`)

```go
// [REFERENCE] New file pkg/optimizer/clip.go.

// ClipByGlobalNorm scales all gradients in `grads` so that the L2 norm of
// their concatenation does not exceed `threshold`. In-place.
func ClipByGlobalNorm[T utils.Float](grads [][]T, threshold T) {
    var sumSq T
    for _, g := range grads {
        for _, v := range g {
            sumSq += v * v
        }
    }
    norm := T(math.Sqrt(float64(sumSq)))
    if norm <= threshold {
        return
    }
    scale := threshold / norm
    for _, g := range grads {
        for i := range g {
            g[i] *= scale
        }
    }
}
```

Invoked by `pkg/nn/train.go` between gradient accumulation and `opt.Step` when
`config.GradClipNorm > 0`.

### 5.10 Step API (`pkg/layer/recurrent/lstm.go` example)

```go
// [REFERENCE] In pkg/layer/recurrent/lstm.go.

// Step advances the LSTM one time step. Uses the internal stepH / stepC
// state, which is updated in place. Use ResetState() between independent
// sequences.
func (l *LSTM[T]) Step(x []T) []T {
    if len(x) != l.InSize {
        panic("recurrent: Step input size mismatch")
    }
    if l.stepH == nil {
        l.stepH = make([]T, l.Hidden)
        l.stepC = make([]T, l.Hidden)
    }
    // Compute 4 gates from x, stepH, biases. Update stepC, stepH in place.
    // Return stepH (caller may copy if it needs to retain it).
    return l.stepH
}

func (l *LSTM[T]) ResetState() {
    if l.stepH != nil { clear(l.stepH) }
    if l.stepC != nil { clear(l.stepC) }
}
```

### 5.11 Error Sentinels (`pkg/utils/errors.go`)

```go
// [REFERENCE] Additions to pkg/utils/errors.go.
var (
    ErrRecurrentShape       = errors.New("recurrent layer shape mismatch")
    ErrRecurrentStepNoInit  = errors.New("recurrent Step called before layer initialised")
    ErrGradClipInvalid      = errors.New("grad clip threshold must be > 0")
)
```

All three wrap the `Compute` category sentinel from `l1-error-taxonomy.md`.

## 6. Implementation Notes

The L1 spec sketches three sub-phases (SimpleRNN baseline → LSTM → GRU). The concrete task
order in PLAN.md / TASKS.md should be:

1. **Phase α** — `utils.Orthogonal` + tests; gate before any cell uses it.
2. **Phase β** — `pkg/layer/recurrent/cell.go` shared helpers (sigmoid/tanh fused, state cache buffers, layer-interface boilerplate).
3. **Phase γ** — `SimpleRNN[T]` Forward + Backward + Init + JSON + Step + tests (finite-diff gradient check).
4. **Phase δ** — `LSTM[T]` same. Largest commit; cross-references Phase γ for the BPTT pattern.
5. **Phase ε** — `GRU[T]` same. Smaller than δ; reuses gate-loop helpers.
6. **Phase ζ** — `LastStep[T]` (stateless, smallest commit).
7. **Phase η** — `pkg/optimizer/clip.go` + `WithGradClipNorm` + `train.go` integration.
8. **Phase θ** — `pkg/nn` options (`WithSimpleRNN` / `WithLSTM` / `WithGRU` / `WithLastStep`) + `compile.go` wiring + integration test (small sentiment-classifier example over a toy dataset).

Phases α and β have no dependencies among themselves; γ, δ, ε must run serial (each builds on
the prior cell-type's gate-loop pattern); ζ and η are independent of γ/δ/ε and can land in
parallel; θ gates on γ/δ/ε complete.

## 7. Drawbacks & Alternatives

- **Alternative: separate `LSTMCell` / `LSTM` (multi-layer wrapper)** — rejected; the project's existing layer abstraction is per-layer not per-cell. Multi-layer stacking is handled by the topology graph (`WithLSTM(...) WithLSTM(...)`).
- **Alternative: pre-allocate fused gate cache across batches** — deferred to a perf-impl amendment; v0.1.0 allocates per Forward call and relies on Go GC. Bench results during Phase δ will tell whether `sync.Pool` is worthwhile.
- **Alternative: stdlib `math.Exp` directly in sigmoid hot loop** — rejected; reuse the existing `activation.Sigmoid[T]` dispatcher to keep the kernel under the AI-Meta documentation regime and the activation registry consistent.
- **Alternative: `WithGradClipNorm` as a `Optimizer` option instead of a top-level Option** — rejected; clipping is a property of the gradient accumulation step, not the optimizer's update rule. Cleaner to keep it on `config` and apply in `train.go`.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[L1-PARENT]` | `.design/main/specifications/l1-recurrent-layers.md` | Parent contract — REC-1..REC-9 invariants are the input to §4 |
| `[LAYER-IFACE]` | `.design/main/specifications/l2-layer-types.md` | `Layer[T]` interface this package satisfies |
| `[CONV-PEER]` | `.design/main/specifications/l2-conv-layers-impl.md` | Reference for `pkg/layer/conv/` layout and option-wiring style |
| `[INIT-IMPL]` | `.design/main/specifications/l2-init-impl.md` | Weight init helpers — extend with `Orthogonal` for `W_hh` |
| `[OPT-IMPL]` | `.design/main/specifications/l2-optimizer-impl.md` | Gradient-clip integration in optimizer step path |
| `[TRAIN-IMPL]` | `.design/main/specifications/l2-training-loop.md` | Backward dispatch — BPTT extends the standard backward |
| `[PERSIST-IMPL]` | `.design/main/specifications/l2-persistence-impl.md` | JSON round-trip for recurrent weights |
| `[CONV-PKG]` | `pkg/layer/conv/` | Sibling package — file layout, var assertions, option naming pattern |
| `[LAYER-PKG]` | `pkg/layer/` | Parent package; new sub-package goes alongside conv/norm |
| `[ERR-PKG]` | `pkg/utils/errors.go` | Three new sentinels per §5.11 |
| `[INIT-PKG]` | `pkg/utils/init.go` | `Orthogonal` helper addition per §5.8 |
| `[OPT-PKG]` | `pkg/optimizer/` | New `clip.go` file per §5.9 |

<!-- Downstream agent instruction: §5.1 package layout is the file blueprint. §5.2/§5.3/§5.4
     struct definitions are the schema — Forward/Backward methods MUST operate on the named
     fields verbatim. §5.8 Orthogonal init is the single most failure-prone area; the test
     suite MUST assert ‖Q·Q^T - I‖_F < 1e-10 (float64) / 1e-5 (float32) before any cell uses it. -->

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-18 | Initial spec authored via `/magic-spec` Spark 1b. `pkg/layer/recurrent/` package with `SimpleRNN[T]` / `LSTM[T]` / `GRU[T]` / `LastStep[T]`. Five new `pkg/nn` options (`WithSimpleRNN` / `WithLSTM` / `WithGRU` / `WithLastStep` / `WithGradClipNorm`), `utils.Orthogonal` init helper, `optimizer.ClipByGlobalNorm`, three new error sentinels. 8-phase implementation plan (α-θ). Promoted Draft → Stable via Trust Mode (MVC: Overview + §4 Invariant Compliance + §5 Detailed Design; no RULES conflicts; no cycles; L1 parent `l1-recurrent-layers.md` Stable v0.1.0). |
