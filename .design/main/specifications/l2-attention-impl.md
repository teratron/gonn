# Attention Mechanism — Go Implementation

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-attention.md

## Overview

Concrete Go realization of [l1-attention.md](l1-attention.md) — attention layers delivered as the
`pkg/layer/attention/` package. Single canonical struct `MultiHeadAttention[T]` covers all three
L1 conceptual variants (`Attention`, `SelfAttention`, `MultiHeadAttention`) via the `NumHeads`
field: `NumHeads = 1` gives the single-head Attention / SelfAttention variants; `NumHeads > 1`
gives multi-head. Two constructors disambiguate intent at call sites (`NewAttention(...)` and
`NewMultiHeadAttention(...)`). The layer satisfies the existing `layer.Layer[T]` interface so the
topology graph treats it uniformly with Dense / Conv / Recurrent layers.

Adds three `pkg/nn/` functional options (`WithAttention`, `WithMultiHeadAttention`,
`WithCausalAttention`), one new optional `MaskedLayer[T]` interface for runtime padding-mask
injection, two new error sentinels in `pkg/utils/errors.go`, and four softmax helpers
(`softmaxRowwise`, `softmaxRowwiseWithMask`, `softmaxBackwardRowwise`,
`softmaxBackwardRowwiseWithMask`) in a shared `cell.go` mirroring the layout of
`pkg/layer/recurrent/cell.go`.

## Related Specifications

- [l1-attention.md](l1-attention.md) — Parent — ATT-1..ATT-10 invariants + ATT-C1..ATT-C8 constraints
- [l2-layer-types.md](l2-layer-types.md) — `Layer[T]` interface this package satisfies
- [l2-recurrent-impl.md](l2-recurrent-impl.md) — Reference for `pkg/layer/recurrent/` layout pattern (sibling sequence package); cell.go helper pattern mirrored here
- [l2-conv-layers-impl.md](l2-conv-layers-impl.md) — Reference for `pkg/layer/conv/` option-wiring style
- [l2-init-impl.md](l2-init-impl.md) — Xavier initialization helper for the four projection matrices
- [l2-training-loop.md](l2-training-loop.md) — Backward pass dispatcher; attention plugs in at the standard layer-backward extension point
- [l2-persistence-impl.md](l2-persistence-impl.md) — JSON round-trip for projection weights per ATT-9
- [l2-regularization-impl.md](l2-regularization-impl.md) — Existing `Dropout[T]` primitive; future amendment wires it as post-softmax attention-weight regularizer (ATT-C7)
- [l2-activation-functions.md](l2-activation-functions.md) — `activation.Softmax[T]` is row-wise unmasked baseline; this package adds masked + backward variants in `cell.go`

## 1. Motivation

L1 fixes the math (ATT-2 / ATT-3 forward equations, ATT-4 multi-head reshape, ATT-5 / ATT-6
masking semantics) and the gradient-flow contract (ATT-7 four-path backward). This spec fixes
the Go specifics:

- **Package path**: `pkg/layer/attention/` — peer to `pkg/layer/conv/`, `pkg/layer/recurrent/`, `pkg/layer/norm/`.
- **Layer-interface conformance**: explicit `var _ layer.Layer[float64] = (*MultiHeadAttention[float64])(nil)` compile-time assertion.
- **Forward cache layout**: `[]T` flat with head-major indexing for per-head Q/K/V (`[NumHeads, SeqLen, Dk]`) and post-softmax weights (`[NumHeads, SeqLen, SeqLen]`).
- **Multi-head reshape**: implemented as index arithmetic on a single flat buffer — no Go slice-of-slice or matrix-library tensor abstraction. Stays stdlib-only per C29.
- **Padding mask**: optional auxiliary input delivered via the new `MaskedLayer[T]` interface — `SetPaddingMask(m []bool)` is called between batches; `Forward` reads internal state. Mirrors the `SetInitialState` pattern from recurrent layers.
- **Softmax backward**: standalone helpers in `cell.go` (NOT inside `activation` package) because the row-wise Jacobian with optional mask differs from `activation.Softmax`'s loss-fused backward path.
- **Option wiring**: three new options propagate the attention layer onto `config.AttentionLayers`, consumed by `compile()` after the Recurrent layers and before the Dense stack.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| ATT-1 (Forward shape) | `Forward` accepts `(SeqLen, Dmodel)` flattened to `[]T` of length `SeqLen·Dmodel`; layer's `inputShape()` exposes `(SeqLen, Dmodel)` to `compile()`; output shape preserved end-to-end |
| ATT-2 (Projection matrices) | `Wq`/`Wk`/`Wv`/`Wo` each `[Dmodel, Dmodel]` row-major flat `[]T`; biases `Bq`/`Bk`/`Bv`/`Bo` each `[Dmodel]`; projections via single GEMM call per matrix in `pkg/layer/attention/attention.go` |
| ATT-3 (Scaled dot-product) | `score = Q · Kᵀ / sqrt(Dk)` in `cell.go` helper `attentionScores(...)`; scaling constant `1.0 / math.Sqrt(float64(Dk))` precomputed once per Forward; softmax via `softmaxRowwise` / `softmaxRowwiseWithMask`; test `TestAttentionScaleFactor` asserts softmax-output entropy `> 0.5 * ln(SeqLen)` for `Dk = 64`, catching missing-scale bugs |
| ATT-4 (Multi-head split) | Head-major flat layout: index formula `[h, t, k] → h*SeqLen*Dk + t*Dk + k`; `splitHeads(buf, numHeads)` and `joinHeads(buf, numHeads)` are pure index arithmetic — no allocation, no copy when `Dmodel = NumHeads * Dk` (always true per ATT-C3); per-head independent scoring loops in `forwardMultiHead` |
| ATT-5 (Causal masking) | When `Causal` field is true, `applyCausalMask(scores, seqLen)` sets `scores[i, j] = -math.Inf(-1)` for `j > i` before softmax; mask is recomputed inline per call (cheap O(SeqLen²) write); construction-time flag — no runtime cost when disabled |
| ATT-6 (Padding masking) | `SetPaddingMask(m []bool)` stores `m` on the layer; `applyPaddingMask(scores, m)` sets `scores[i, j] = -math.Inf(-1)` for `m[j] = false`, broadcasting over the query axis; both masks compose additively (sequential `applyCausalMask` then `applyPaddingMask` writes); `Forward` reads `padMask` from layer state and resets it after consumption to enforce per-call semantics |
| ATT-7 (Backward correctness) | `Backward` walks the four-path decomposition in `cell.go` helper `softmaxBackwardRowwiseWithMask`: (1) upstream → `Wo` + `Bo` + per-head dA; (2) `dA = dOut · Vᵀ`, `dV = Aᵀ · dOut`; (3) softmax-backward Jacobian: `dScores[i, :] = (dA[i, :] - <dA[i, :], A[i, :]>) ⊙ A[i, :]`; (4) `dQ = dScores · K / sqrt(Dk)`, `dK = dScoresᵀ · Q / sqrt(Dk)`; finite-diff gradient check in `attention_test.go` validates within `1e-4` tolerance for `T=float64` |
| ATT-8 (Weight initialization) | `Init(rng)` calls `utils.Xavier[T](rng, Dmodel)` for each of `Wq`/`Wk`/`Wv`/`Wo`; biases initialised to zero via `clear()`; test `TestAttentionInitStats` asserts each projection matrix has Frobenius norm within `[0.5·sqrt(Dmodel), 1.5·sqrt(Dmodel)]` |
| ATT-9 (Persistence) | `MarshalJSON` serialises `{Type, SeqLen, Dmodel, NumHeads, Causal, Wq, Wk, Wv, Wo, Bq, Bk, Bv, Bo}`; `UnmarshalJSON` validates length parity (e.g. `len(Wq) == Dmodel*Dmodel`) and `NumHeads` divisibility of `Dmodel` per ATT-C3; padding mask explicitly NOT serialised (runtime-only) |
| ATT-10 (Layer interface compatibility) | Compile-time assertion `var _ layer.Layer[float64] = (*MultiHeadAttention[float64])(nil)` in `multihead.go`; `Forward`/`Backward`/`Init`/`MarshalJSON` signatures match Dense / Conv / Recurrent; optional `MaskedLayer[T]` interface with `SetPaddingMask([]bool)` is asserted separately — only this layer implements it |

## 5. Detailed Design

### 5.1 Package Layout

```text
pkg/layer/attention/
├── doc.go              // package documentation + AI-Meta block
├── cell.go             // shared helpers: softmax row-wise (+masked) forward & backward,
│                       //                  scaling factor precompute, head reshape arithmetic
├── multihead.go        // MultiHeadAttention[T] — single struct covering all three L1 variants
├── masked_layer.go     // MaskedLayer[T] interface + SetPaddingMask wiring on MultiHeadAttention
├── attention_test.go   // single-head Forward / Backward / Init / JSON; finite-diff gradient check
├── multihead_test.go   // multi-head Forward / Backward; head-split round-trip; reshape arithmetic
├── mask_test.go        // causal mask vs unmasked diff; padding mask broadcast; both-masks composition
└── cell_test.go        // softmax helpers row-stochasticity + numerical stability (-1e6 scores)
```

### 5.2 MultiHeadAttention Struct

```go
// [REFERENCE] In pkg/layer/attention/multihead.go.
package attention

type MultiHeadAttention[T utils.Float] struct {
    SeqLen   int
    Dmodel   int
    NumHeads int
    Dk       int  // Dmodel / NumHeads — cached
    Causal   bool

    // Projection matrices (each [Dmodel, Dmodel] row-major flat).
    Wq, Wk, Wv, Wo []T
    Bq, Bk, Bv, Bo []T

    // Forward cache — populated by Forward, consumed by Backward.
    lastInput     []T // [SeqLen, Dmodel]
    lastQ, lastK, lastV []T // each [NumHeads, SeqLen, Dk] head-major flat
    lastWeights   []T // [NumHeads, SeqLen, SeqLen] post-softmax

    // Per-call padding mask state — set by SetPaddingMask, cleared after Forward.
    padMask []bool // [SeqLen]; nil = all valid

    // Gradient slots — pre-allocated; zeroed at the start of each Backward.
    gradWq, gradWk, gradWv, gradWo []T
    gradBq, gradBk, gradBv, gradBo []T
    gradX []T

    // Precomputed scaling constant 1.0 / sqrt(Dk).
    scale T
}

var _ layer.Layer[float64] = (*MultiHeadAttention[float64])(nil)
```

### 5.3 MaskedLayer Interface

```go
// [REFERENCE] In pkg/layer/attention/masked_layer.go.
package attention

// MaskedLayer is an optional interface for layers that accept a per-forward
// boolean padding mask. Only attention layers implement it. Callers that
// supply a padding mask MUST invoke SetPaddingMask between batches; the
// layer consumes and clears the mask in Forward to enforce per-call semantics.
type MaskedLayer[T utils.Float] interface {
    layer.Layer[T]
    SetPaddingMask(m []bool)
}

var _ MaskedLayer[float64] = (*MultiHeadAttention[float64])(nil)

func (m *MultiHeadAttention[T]) SetPaddingMask(mask []bool) {
    if len(mask) != m.SeqLen {
        panic("attention: padding mask length mismatch")
    }
    m.padMask = mask
}
```

### 5.4 Constructors

```go
// [REFERENCE] In pkg/layer/attention/multihead.go.

// NewAttention constructs a single-head attention layer (NumHeads = 1).
// Equivalent to NewMultiHeadAttention(seqLen, dmodel, 1, causal).
func NewAttention[T utils.Float](seqLen, dmodel int, causal bool) *MultiHeadAttention[T] {
    return NewMultiHeadAttention[T](seqLen, dmodel, 1, causal)
}

// NewMultiHeadAttention constructs a multi-head attention layer.
// numHeads MUST divide dmodel evenly (ATT-C3); a non-divisible pair
// triggers ErrAttentionHeadsMismatch.
func NewMultiHeadAttention[T utils.Float](seqLen, dmodel, numHeads int, causal bool) *MultiHeadAttention[T] {
    if dmodel%numHeads != 0 {
        panic(fmt.Errorf("attention: %w (Dmodel=%d, NumHeads=%d)", utils.ErrAttentionHeadsMismatch, dmodel, numHeads))
    }
    dk := dmodel / numHeads
    return &MultiHeadAttention[T]{
        SeqLen: seqLen, Dmodel: dmodel, NumHeads: numHeads, Dk: dk, Causal: causal,
        Wq: make([]T, dmodel*dmodel), Wk: make([]T, dmodel*dmodel),
        Wv: make([]T, dmodel*dmodel), Wo: make([]T, dmodel*dmodel),
        Bq: make([]T, dmodel), Bk: make([]T, dmodel),
        Bv: make([]T, dmodel), Bo: make([]T, dmodel),
        scale: T(1.0 / math.Sqrt(float64(dk))),
    }
}
```

### 5.5 Cell Helpers (`pkg/layer/attention/cell.go`)

```go
// [REFERENCE] In pkg/layer/attention/cell.go.
package attention

// softmaxRowwise applies softmax over the last axis of an [N, K] matrix flattened to []T.
// Numerically stable: subtracts the row max before exp.
func softmaxRowwise[T utils.Float](scores []T, n, k int) { /* ... */ }

// softmaxRowwiseWithMask applies softmax row-wise, treating positions where mask[j] = false
// as -Inf (effectively zero softmax weight). When mask == nil, behaves like softmaxRowwise.
func softmaxRowwiseWithMask[T utils.Float](scores []T, n, k int, mask []bool) { /* ... */ }

// softmaxBackwardRowwise computes the Jacobian-vector product for the row-wise softmax:
//   dScores[i, :] = (dA[i, :] - <dA[i, :], A[i, :]>) ⊙ A[i, :]
// Writes into dScores in place.
func softmaxBackwardRowwise[T utils.Float](dA, A, dScores []T, n, k int) { /* ... */ }

// softmaxBackwardRowwiseWithMask handles masked positions: their dScores contribution is zero
// because the corresponding A[i, j] is zero. Single-pass over [n, k].
func softmaxBackwardRowwiseWithMask[T utils.Float](dA, A, dScores []T, n, k int, mask []bool) { /* ... */ }

// attentionScores fills scores[h, i, j] = Q[h, i, :] · K[h, j, :] * scale.
// Head-major layout; scale precomputed once per Forward call.
func attentionScores[T utils.Float](Q, K, scores []T, numHeads, seqLen, dk int, scale T) { /* ... */ }

// applyCausalMask sets scores[h, i, j] = -Inf for all j > i.
// Operates in place; numHeads × seqLen² writes total.
func applyCausalMask[T utils.Float](scores []T, numHeads, seqLen int) { /* ... */ }
```

Helpers are package-private; their signatures are stable across attention layer types and
correspond directly to ATT-3 / ATT-5 / ATT-6 / ATT-7 invariants.

### 5.6 Functional Options (`pkg/nn/options.go`)

```go
// [REFERENCE] Additions to pkg/nn/options.go.

func WithAttention[T utils.Float](seqLen, dmodel int) Option[T] {
    return func(c *config[T]) {
        c.AttentionLayers = append(c.AttentionLayers, attention.NewAttention[T](seqLen, dmodel, false))
    }
}

func WithMultiHeadAttention[T utils.Float](seqLen, dmodel, numHeads int) Option[T] {
    return func(c *config[T]) {
        c.AttentionLayers = append(c.AttentionLayers, attention.NewMultiHeadAttention[T](seqLen, dmodel, numHeads, false))
    }
}

// WithCausalAttention sets the Causal flag on the most recently added attention layer.
// Order matters: must be invoked AFTER the WithAttention / WithMultiHeadAttention it modifies.
func WithCausalAttention[T utils.Float]() Option[T] {
    return func(c *config[T]) {
        if n := len(c.AttentionLayers); n > 0 {
            if mha, ok := c.AttentionLayers[n-1].(*attention.MultiHeadAttention[T]); ok {
                mha.Causal = true
            }
        }
    }
}
```

`config[T]` gains one field: `AttentionLayers []layer.Layer[T]` (parallel to `RecurrentLayers`).
The causal flag is a post-hoc modifier rather than a constructor parameter to keep the option
arity consistent with `WithSimpleRNN` / `WithLSTM` / `WithGRU`.

### 5.7 Compile Wiring (`pkg/nn/compile.go`)

The compile order becomes:

```text
[Input] → [Conv2D prefix] → [Conv1D prefix] → [Recurrent layers] → [Recurrent tail]
       → [Attention layers] → [Dense stack] → [Output]
```

`setupAttentionShapes()` runs after the recurrent shape walk and before the Dense walk. It
verifies that the upstream shape is `(SeqLen, Dmodel)` and matches each attention layer's
declared parameters. Shape preservation (ATT-1) means downstream attention layers can chain
without intermediate shape adapters.

### 5.8 Error Sentinels (`pkg/utils/errors.go`)

```go
// [REFERENCE] Additions to pkg/utils/errors.go.
var (
    ErrAttentionHeadsMismatch = errors.New("attention: Dmodel must be divisible by NumHeads")
    ErrAttentionMaskLength    = errors.New("attention: padding mask length must equal SeqLen")
)
```

Both wrap the `Compute` category sentinel from `l1-error-taxonomy.md`.

### 5.9 Padding Mask Usage Pattern

```go
// [REFERENCE] Caller-side pattern.
//
//   net := nn.New(...,
//       nn.WithMultiHeadAttention[float64](seqLen, 128, 8),
//       nn.WithCausalAttention[float64](),
//   )
//
//   for batch := range dataset {
//       // Find the attention layer and supply per-batch padding mask.
//       if mha, ok := net.LayerByIndex(idx).(attention.MaskedLayer[float64]); ok {
//           mha.SetPaddingMask(batch.PadMask)
//       }
//       net.Forward(batch.Input)
//   }
```

The compile step exposes attention layers through `Network[T].LayerByIndex(int)` (already
exists for layer introspection in `pkg/network/`). When the user does NOT supply a padding mask,
`Forward` proceeds with `padMask == nil` and the masked-softmax helper falls back to the
unmasked path — zero overhead in the no-mask case.

### 5.10 Dropout Integration (Deferred to Future Amendment)

ATT-C7 (post-softmax attention-weight dropout) is opt-in and deferred to a v0.2 amendment. The
`MultiHeadAttention[T]` struct reserves a `Dropout *regularizer.Dropout[T]` field stub; v0.1
leaves it `nil` and the Forward path skips the dropout step. Adding it later is a non-breaking
change — current persistence schema unchanged because the field is nil-able and not serialised
when nil (per `omitempty`).

## 6. Implementation Notes

The L1 spec sketches three sub-phases (single-head baseline → multi-head split → mask
integration). The concrete task order in PLAN.md / TASKS.md should be:

1. **Phase α — Softmax helpers + masked softmax in `cell.go`**: `softmaxRowwise`, `softmaxRowwiseWithMask`, `softmaxBackwardRowwise`, `softmaxBackwardRowwiseWithMask`, plus `attentionScores` and `applyCausalMask`. Tests: numerical stability under `±1e6` score range, row-stochasticity under mask, finite-difference gradient validation. Gate before any attention layer uses them.

2. **Phase β — `MultiHeadAttention[T]` single-head baseline** (`NumHeads = 1`, `Causal = false`, no padding mask): full Forward / Backward / Init / JSON. Finite-diff gradient check on all four projection matrices and biases. Test `TestAttentionScaleFactor` asserts softmax entropy `> 0.5 * ln(SeqLen)` at `Dk = 64`.

3. **Phase γ — Multi-head reshape**: extend Forward / Backward to `NumHeads > 1`. Verify `splitHeads + joinHeads` is an identity transform. Test that single-head and multi-head with identity projection matrices produce numerically equal outputs on the same input (subject to floating-point rounding).

4. **Phase δ — Mask integration**: `Causal` construction flag + `SetPaddingMask` runtime input + `MaskedLayer[T]` interface. Tests:
   - Causal mask: position `i` cannot attend to `j > i` — assert by zeroing future inputs and checking output unchanged at position `i`.
   - Padding mask: masked-vs-unmasked gradient equality on unmasked positions.
   - Both masks composed: row-stochasticity of `A` preserved.

5. **Phase ε — `pkg/nn` options + compile wiring + integration test**: `WithAttention` / `WithMultiHeadAttention` / `WithCausalAttention` options. `setupAttentionShapes` in `compile.go`. Small synthetic integration test: 2-layer multi-head encoder + Dense + Output on a toy sequence-classification task (e.g. "is this random sequence monotone?"). Convergence in a known number of epochs is the smoke test.

Phases α and β must run serial (β depends on α's helpers). γ extends β's Forward/Backward — also serial. δ depends on γ (mask helpers integrate into the multi-head Forward path). ε gates on δ complete.

## 7. Drawbacks & Alternatives

- **Alternative: separate `SingleHeadAttention[T]` and `MultiHeadAttention[T]` struct types** — rejected. Code duplication outweighs the marginal naming clarity. `NumHeads = 1` is a perfectly valid single-head config and the constructor `NewAttention` provides the naming clarity at call sites.
- **Alternative: padding mask as a `Forward` extra argument** — rejected. The project's `Layer[T]` interface signature is `Forward(input []T) []T`; adding a second argument would break the interface and every existing layer. The `MaskedLayer[T]` opt-in extension via `SetPaddingMask` is non-breaking.
- **Alternative: store the causal mask as a precomputed `[]bool` per attention layer** — rejected for v0.1. The mask is `SeqLen²` size; precomputing is faster but doubles memory per layer. v0.1 recomputes inline (cheap `O(SeqLen²)` writes); a future perf-impl amendment can add precomputation as an opt-in flag when `SeqLen > 512`.
- **Alternative: fuse softmax-backward into a single GEMM-shaped helper** — deferred. The current row-wise loop is `O(SeqLen²)` and matches the forward complexity. Flash-attention-style fused kernels are a future L2 amendment cross-cutting with `l2-backend-gpu.md`.
- **Alternative: reuse `activation.Softmax[T]` for the row-wise softmax** — rejected. `activation.Softmax` is designed for the OUTPUT layer in a classifier (single vector, often fused with cross-entropy loss in backward). Attention needs row-wise on `(SeqLen, SeqLen)` with masking; the API impedance mismatch is large enough that a dedicated `cell.go` helper is cleaner than a wrapper.
- **Alternative: cross-attention support in v0.1** — out of scope. Requires Q-source and K/V-source to come from separate `Forward` inputs, which doesn't fit the single-input `Layer[T]` interface. Future amendment will introduce a `BiInputLayer[T]` extension interface or a paired-layer pattern.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[L1-PARENT]` | `.design/main/specifications/l1-attention.md` | Parent contract — ATT-1..ATT-10 invariants are the input to §4 |
| `[LAYER-IFACE]` | `.design/main/specifications/l2-layer-types.md` | `Layer[T]` interface this package satisfies |
| `[REC-PEER]` | `.design/main/specifications/l2-recurrent-impl.md` | Sibling sequence package — file layout, cell.go pattern, option-wiring style |
| `[CONV-PEER]` | `.design/main/specifications/l2-conv-layers-impl.md` | Sibling layer package — option-wiring style |
| `[INIT-IMPL]` | `.design/main/specifications/l2-init-impl.md` | Xavier init helper for the four projection matrices |
| `[TRAIN-IMPL]` | `.design/main/specifications/l2-training-loop.md` | Backward dispatch — attention extends the standard backward |
| `[PERSIST-IMPL]` | `.design/main/specifications/l2-persistence-impl.md` | JSON round-trip for projection weights per ATT-9 |
| `[ACT-IMPL]` | `.design/main/specifications/l2-activation-functions.md` | `activation.Softmax[T]` baseline — separate from this package's row-wise + masked variants |
| `[REG-IMPL]` | `.design/main/specifications/l2-regularization-impl.md` | Future amendment will wire `regularizer.Dropout[T]` as post-softmax attention-weight dropout (ATT-C7) |
| `[REC-PKG]` | `pkg/layer/recurrent/` | Sibling package — cell.go pattern, var assertions, option naming pattern |
| `[LAYER-PKG]` | `pkg/layer/` | Parent package; new sub-package goes alongside recurrent/conv/norm |
| `[ERR-PKG]` | `pkg/utils/errors.go` | Two new sentinels per §5.8 |

<!-- Downstream agent instruction: §5.1 package layout is the file blueprint. §5.2 struct
     definition is the schema — Forward/Backward methods MUST operate on the named fields
     verbatim. §5.5 cell helpers are the contract for ATT-3 / ATT-5 / ATT-6 / ATT-7; their
     unit tests gate the entire attention layer. §5.4 NewMultiHeadAttention's divisibility
     panic is the canonical place to surface ATT-C3 violations. The Dropout field stub in
     §5.10 is intentional — leave nil for v0.1; the v0.2 amendment fills it. -->

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-18 | Initial spec authored via `/magic-spec` Blank Trigger (follow-up to l1-attention v0.1.0). `pkg/layer/attention/` package with single `MultiHeadAttention[T]` struct covering all three L1 conceptual variants via `NumHeads` field; `NewAttention` / `NewMultiHeadAttention` constructors. Three new `pkg/nn` options (`WithAttention` / `WithMultiHeadAttention` / `WithCausalAttention`), `MaskedLayer[T]` optional interface for runtime padding-mask injection, two new error sentinels (`ErrAttentionHeadsMismatch`, `ErrAttentionMaskLength`), four softmax helpers in `cell.go` (row-wise forward/backward, masked variants), inline causal-mask helper. 5-phase implementation plan (α-ε). Invariant Compliance table covers ATT-1..ATT-10 with concrete file/line-level mapping; ATT-3 scale-factor and ATT-7 four-path backward flagged as highest-risk test gates. Promoted Draft → Stable via Trust Mode (C9): MVC satisfied (Overview + §4 Invariant Compliance + §5 Detailed Design); no RULES.md conflicts; no circular dependencies; L1 parent `l1-attention.md` Stable v0.1.0. |
