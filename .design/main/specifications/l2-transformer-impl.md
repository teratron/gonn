# Transformer Block — Go Implementation

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-transformer-block.md

## Overview

Concrete Go realization of [l1-transformer-block.md](l1-transformer-block.md) — Transformer blocks delivered as the `pkg/layer/transformer/` package. Three exported types: `EncoderBlock[T]` (bidirectional self-attention + FFN + two residuals + two LayerNorms), `DecoderBlock[T]` (same wiring with causal masking forwarded to the inner MHA), and `Stack[T]` (N homogeneous blocks routed sequentially). All three satisfy `layer.Layer[T]` and forward optional padding masks via `MaskedLayer[T]`. The blocks own their children (MultiHeadAttention + LayerNorm × 2 + Dense × 2) — owners delegate persistence to children per TRANS-9 rather than re-serializing their parameters.

Adds four `pkg/nn/` functional options (`WithEncoderBlock`, `WithDecoderBlock`, `WithEncoderStack`, `WithDecoderStack`), one config struct `TransformerConfig[T]` aggregating the shared construction parameters, and a thin `Block[T]` private interface in `pkg/layer/transformer/block.go` so `Stack[T]` can hold encoder or decoder children uniformly without exposing a union type.

## Related Specifications

- [l1-transformer-block.md](l1-transformer-block.md) — Parent — TRANS-1..TRANS-10 invariants + TRANS-C1..TRANS-C8 constraints
- [l2-layer-types.md](l2-layer-types.md) — `Layer[T]` interface this package satisfies; `Dense[T]` reused for FFN
- [l2-attention-impl.md](l2-attention-impl.md) — `MultiHeadAttention[T]` reused as the inner attention primitive; `MaskedLayer[T]` extension forwarded
- [l2-normalization-impl.md](l2-normalization-impl.md) — `LayerNorm[T]` reused; BatchNorm and GroupNorm explicitly rejected per TRANS-C5
- [l2-activation-functions.md](l2-activation-functions.md) — ReLU default for FFN; Gelu deferred (TRANS-C4)
- [l2-regularization-impl.md](l2-regularization-impl.md) — `Dropout[T]` reused for three placements per TRANS-C7
- [l2-embedding-impl.md](l2-embedding-impl.md) — Upstream producer; `EmbeddingStack[T]` output feeds Transformer Stack
- [l2-init-impl.md](l2-init-impl.md) — Xavier on FFN Dense matrices; inherits attention init
- [l2-persistence-impl.md](l2-persistence-impl.md) — JSON envelope; composite block delegates to children per TRANS-9
- [l2-training-loop.md](l2-training-loop.md) — Backward dispatcher; composite block routes upstream through its children in reverse order

## 1. Motivation

L1 fixes the math (TRANS-2/3 forward composition, TRANS-5 FFN structure) and the contracts (TRANS-6 residual, TRANS-8 backward, TRANS-9 child-delegated persistence). This spec fixes the Go specifics:

- **Package path**: `pkg/layer/transformer/` — peer to `pkg/layer/attention/`, `pkg/layer/conv/`, `pkg/layer/recurrent/`, `pkg/layer/norm/`, `pkg/layer/embedding/`.
- **Child-ownership pattern**: each block holds typed children (`*attention.MultiHeadAttention[T]`, `*norm.LayerNorm[T]`, `*dense.Dense[T]`) rather than `[]Layer[T]`. Typed ownership gives zero-cost access in the hot path and explicit dependency declarations.
- **Block[T] internal interface**: `Stack[T]` holds `[]Block[T]` where `Block[T]` is a package-private interface with `Forward` / `Backward` / `Children()` methods. Lets Stack be uniform over Encoder and Decoder without exposing a union type to the public API.
- **Mask forwarding**: `EncoderBlock[T]` and `DecoderBlock[T]` implement `MaskedLayer[T]` (from `l2-attention-impl.md`) — `SetPaddingMask(m)` simply forwards `m` to the inner MHA's `SetPaddingMask`. The intermediate LayerNorm and FFN need no mask awareness per TRANS-§4.6.
- **Persistence delegation**: block `MarshalJSON` writes `{Type, Config, Children: {Attn, Norm1, Norm2, FFN1, FFN2}}` where the `Children` map holds each child's pre-marshalled JSON object. `UnmarshalJSON` validates the Type tag and dispatches into each child's `UnmarshalJSON`. No field-level re-marshalling.
- **Stack persistence**: `Stack[T].MarshalJSON` writes `{Type, Config, Blocks: [...]}` where each block is itself pre-marshalled. Round-trip restores the topology exactly.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| TRANS-1 (Shape preservation) | Each block's `Forward([]T) ([]T, error)` accepts `[]T` of length `SeqLen × Dmodel` and returns the same length; `Stack[T].Forward` chains N blocks via `out, err = blocks[i].Forward(out)` loop |
| TRANS-2 (Encoder post-norm) | When `cfg.PreNorm == false`: `EncoderBlock[T].Forward` computes the chain attn→drop→add→norm₁→ffn→drop→add→norm₂; intermediate buffers reused via a `bufPair` field on the block to avoid per-call allocation (PERF-4) |
| TRANS-3 (Encoder pre-norm) | When `cfg.PreNorm == true`: forward computes norm₁(X)→attn→drop→add(X)→norm₂(Z)→ffn→drop→add(Z); residual buffer kept as a separate `[]T` so the add at step 3 sees the un-normalized input |
| TRANS-4 (Decoder forward) | `DecoderBlock[T]` constructor calls `attention.NewMultiHeadAttention(...)` with `Causal: true`; otherwise identical to EncoderBlock. The Causal flag is part of the inner MHA's persisted config so JSON round-trip preserves the distinction |
| TRANS-5 (FFN structure) | Two `*dense.Dense[T]` children: `FFN1` with shape `(Dmodel → Dff)`, `FFN2` with shape `(Dff → Dmodel)`; activation enum `cfg.Activation` selects from existing `pkg/activation/` dispatcher; default `activation.ReLU` per TRANS-C4 |
| TRANS-6 (Residual unmodified) | `addInPlace[T](dst, src []T)` helper in `block.go` writes `dst[i] += src[i]` — no scaling, no clipping, no normalization; test `TestResidualIdentity` asserts that with attention zero-weights initialized and FFN zero-weights initialized, block output equals input bit-exactly (T=float32 / float64) |
| TRANS-7 (Stack composition) | `Stack[T]` holds `[]Block[T]` of length N; Forward chains sequentially; constructor takes `cfg TransformerConfig[T]` + `N int` and creates N independent block instances with shared config but unique weights via independent `Init(rng)` calls (each block sees a different RNG state) |
| TRANS-8 (Backward correctness) | `Backward([]T) ([]T, error)` routes upstream gradient through children in reverse order (norm₂ → ffn → drop → residual split → norm₁ → attn → drop → residual split); finite-diff gradient check `TestBlockGradient` validates within `1e-4` tolerance on `Dmodel=8, NumHeads=2, Dff=16, SeqLen=4, T=float64`; `Stack[T].Backward` chains N block.Backward in reverse |
| TRANS-9 (Persistence delegation) | `MarshalJSON` writes `{Type:"encoder", Config:{Dmodel,NumHeads,Dff,Causal,PreNorm,DropoutRate,Activation}, Attn, Norm1, Norm2, FFN1, FFN2}` where each named child is its own pre-marshalled object; `UnmarshalJSON` validates Type tag and dispatches into each child's `UnmarshalJSON` — no field-level remarshalling; `Stack[T]` writes `{Type:"stack", Config, Mode:"encoder"\|"decoder", Blocks:[...]}` |
| TRANS-10 (Layer interface compatibility) | Compile-time assertions: `var _ layer.Layer[float64] = (*EncoderBlock[float64])(nil)`, same for Decoder and Stack; all three also implement `attention.MaskedLayer[T]` via a `SetPaddingMask(m []bool)` method that forwards to the inner MHA child (Stack forwards to each block which forwards to its inner MHA) |

## 5. Detailed Design

### 5.1 Package Layout

```text
pkg/layer/transformer/
├── doc.go              // package documentation + AI-Meta block
├── config.go           // TransformerConfig[T] struct + ActivationKind enum
├── block.go            // Block[T] private interface; addInPlace helper; child-construction helpers
├── encoder.go          // EncoderBlock[T] — bidirectional self-attention + FFN composition
├── decoder.go          // DecoderBlock[T] — causal self-attention + FFN composition
├── stack.go            // Stack[T] — N homogeneous blocks routed sequentially; Mode enum
├── encoder_test.go     // EncoderBlock forward/backward; pre-norm vs post-norm equivalence (Dropout=0); JSON round-trip
├── decoder_test.go     // DecoderBlock forward/backward; causal mask propagation; JSON round-trip
├── stack_test.go       // 6-block stack forward/backward; parameter count matches §4.4; copy-task convergence
├── residual_test.go    // TRANS-6 residual identity test (zero-weight attn + zero-weight FFN → output=input)
└── prenorm_test.go     // TRANS-3 pre-norm forward equivalence on independent random init
```

### 5.2 TransformerConfig Struct

```go
// [REFERENCE] In pkg/layer/transformer/config.go.
package transformer

import (
    "github.com/teratron/gonn/pkg/activation"
    "github.com/teratron/gonn/pkg/utils"
)

// TransformerConfig groups the construction parameters shared across encoder
// and decoder blocks. Same config drives every block in a homogeneous Stack.
//
// AI-Meta:
// - Purpose: Capture the architectural hyperparameters of a single Transformer block.
// - Usage: Pass to NewEncoderBlock / NewDecoderBlock / NewStack constructors.
// - Stability: Experimental.
type TransformerConfig[T utils.Float] struct {
    SeqLen      int  // sequence length consumed by the inner MHA
    Dmodel      int  // model / hidden dimension; preserved end-to-end
    NumHeads    int  // attention head count; Dmodel % NumHeads == 0 per ATT-C3
    Dff         int  // FFN hidden dimension; canonical default 4 * Dmodel
    PreNorm     bool // pre-norm if true (TRANS-3), post-norm otherwise (TRANS-2)
    DropoutRate T    // shared rate for the three Dropout positions per TRANS-C7
    Activation  activation.Kind  // FFN activation; default activation.ReLU per TRANS-C4
}
```

### 5.3 EncoderBlock Struct

```go
// [REFERENCE] In pkg/layer/transformer/encoder.go.
package transformer

import (
    "github.com/teratron/gonn/pkg/layer"
    "github.com/teratron/gonn/pkg/layer/attention"
    "github.com/teratron/gonn/pkg/layer/dense"
    "github.com/teratron/gonn/pkg/layer/norm"
    "github.com/teratron/gonn/pkg/layer/regularizer"
)

type EncoderBlock[T utils.Float] struct {
    Cfg TransformerConfig[T]

    Attn   *attention.MultiHeadAttention[T]  // bidirectional (Causal=false)
    Norm1  *norm.LayerNorm[T]
    Norm2  *norm.LayerNorm[T]
    FFN1   *dense.Dense[T]                   // Dmodel -> Dff
    FFN2   *dense.Dense[T]                   // Dff -> Dmodel
    Drop1  *regularizer.Dropout[T]           // post-attention
    Drop2  *regularizer.Dropout[T]           // post-FFN
    // Attention-weight dropout (ATT-C7 position) lives inside Attn, gated via
    // Cfg.DropoutRate when Attn is constructed.

    // Forward cache — reused across calls to avoid allocations (PERF-4).
    bufAttn []T  // attention output
    bufZ    []T  // post-first-residual (post-norm) or post-norm1 (pre-norm)
    bufF    []T  // FFN output
}

var _ layer.Layer[float32] = (*EncoderBlock[float32])(nil)
var _ layer.Layer[float64] = (*EncoderBlock[float64])(nil)
var _ attention.MaskedLayer[float32] = (*EncoderBlock[float32])(nil)
var _ attention.MaskedLayer[float64] = (*EncoderBlock[float64])(nil)

// SetPaddingMask forwards the per-call padding mask to the inner MHA. The mask
// affects only the attention step per TRANS-§4.6; LayerNorm and FFN are mask-agnostic.
//
// AI-Meta:
// - Purpose: Propagate the padding mask to the inner attention layer per call.
// - Usage: Called between batches when input contains padding positions.
// - Concurrency: SingleGoroutine.
// - Stability: Experimental.
func (e *EncoderBlock[T]) SetPaddingMask(m []bool) {
    e.Attn.SetPaddingMask(m)
}
```

### 5.4 DecoderBlock Struct

`DecoderBlock[T]` is structurally identical to `EncoderBlock[T]` — the only delta is the constructor:

```go
// [REFERENCE] In pkg/layer/transformer/decoder.go.

// NewDecoderBlock constructs a TransformerDecoderBlock — same wiring as
// EncoderBlock with causal masking enabled in the inner MultiHeadAttention.
func NewDecoderBlock[T utils.Float](cfg TransformerConfig[T]) *DecoderBlock[T] {
    // Build inner MHA with Causal: true; all other child construction identical to EncoderBlock.
    attn := attention.NewMultiHeadAttention[T](
        cfg.SeqLen, cfg.Dmodel, cfg.NumHeads,
        attention.WithCausal(true),
    )
    // ... build Norm1, Norm2, FFN1, FFN2, Drop1, Drop2 identically.
}
```

The L2 implementation MAY consolidate `EncoderBlock` and `DecoderBlock` into a single struct with a `Causal bool` flag (matching the `PositionalEncoding[T]` / `Mode` and `MultiHeadAttention[T]` / `NumHeads` precedent). v0.1 keeps them as two named types for code clarity; consolidation deferred to a future refactor amendment.

### 5.5 Block Interface (Internal)

```go
// [REFERENCE] In pkg/layer/transformer/block.go.
package transformer

// Block is the private interface that Stack[T] holds. EncoderBlock[T] and
// DecoderBlock[T] both satisfy it. Stack does NOT expose this interface to
// the public API — users construct stacks via NewStack which takes the Mode enum.
//
// AI-Meta:
// - Purpose: Allow Stack to hold encoder and decoder blocks uniformly.
// - Stability: Internal.
type Block[T utils.Float] interface {
    layer.Layer[T]
    attention.MaskedLayer[T]
    children() []layer.Layer[T]  // for the Stack's gradient routing
}
```

### 5.6 Stack Struct

```go
// [REFERENCE] In pkg/layer/transformer/stack.go.
package transformer

type Mode uint8

const (
    EncoderMode Mode = iota
    DecoderMode
)

type Stack[T utils.Float] struct {
    Cfg    TransformerConfig[T]
    Mode   Mode
    Blocks []Block[T]   // N independent block instances
}

var _ layer.Layer[float32] = (*Stack[float32])(nil)
var _ layer.Layer[float64] = (*Stack[float64])(nil)
var _ attention.MaskedLayer[float32] = (*Stack[float32])(nil)
var _ attention.MaskedLayer[float64] = (*Stack[float64])(nil)

// NewStack builds N independent blocks under the same Cfg. Mode selects encoder
// or decoder construction. Each block's Init is called with an independent RNG
// fork so weights are unique despite shared config.
//
// AI-Meta:
// - Purpose: Build a homogeneous stack of N Transformer blocks.
// - Usage: Pass to compile() via WithEncoderStack / WithDecoderStack option.
// - Stability: Experimental.
func NewStack[T utils.Float](cfg TransformerConfig[T], mode Mode, n int) *Stack[T] { /* ... */ }

// SetPaddingMask forwards to every child block, which forwards to its inner MHA.
func (s *Stack[T]) SetPaddingMask(m []bool) {
    for _, b := range s.Blocks {
        b.SetPaddingMask(m)
    }
}
```

### 5.7 Forward Pre-Norm Implementation Sketch

```go
// [REFERENCE] In pkg/layer/transformer/encoder.go — pre-norm branch of Forward.
func (e *EncoderBlock[T]) forwardPreNorm(x []T) ([]T, error) {
    n := len(x)

    // Z1 = LayerNorm1(X)
    z1, err := e.Norm1.Forward(x)
    if err != nil { return nil, fmt.Errorf("transformer encoder Norm1: %w", err) }

    // A = MHA(Z1)
    a, err := e.Attn.Forward(z1)
    if err != nil { return nil, fmt.Errorf("transformer encoder Attn: %w", err) }

    // A' = Dropout1(A)
    aDrop, err := e.Drop1.Forward(a)
    if err != nil { return nil, fmt.Errorf("transformer encoder Drop1: %w", err) }

    // Z = X + A'  (residual; reuses bufZ to avoid allocation)
    if cap(e.bufZ) < n { e.bufZ = make([]T, n) }
    e.bufZ = e.bufZ[:n]
    for i := 0; i < n; i++ {
        e.bufZ[i] = x[i] + aDrop[i]
    }

    // Z2 = LayerNorm2(Z)
    z2, err := e.Norm2.Forward(e.bufZ)
    if err != nil { return nil, fmt.Errorf("transformer encoder Norm2: %w", err) }

    // F = FFN(Z2) = FFN2(Activation(FFN1(Z2)))
    h, err := e.FFN1.Forward(z2)
    if err != nil { return nil, fmt.Errorf("transformer encoder FFN1: %w", err) }
    activation.ApplyInPlace(e.Cfg.Activation, h)
    f, err := e.FFN2.Forward(h)
    if err != nil { return nil, fmt.Errorf("transformer encoder FFN2: %w", err) }

    // F' = Dropout2(F)
    fDrop, err := e.Drop2.Forward(f)
    if err != nil { return nil, fmt.Errorf("transformer encoder Drop2: %w", err) }

    // Y = Z + F'  (residual)
    if cap(e.bufF) < n { e.bufF = make([]T, n) }
    e.bufF = e.bufF[:n]
    for i := 0; i < n; i++ {
        e.bufF[i] = e.bufZ[i] + fDrop[i]
    }

    return e.bufF, nil
}
```

(Post-norm branch is structurally similar — Norm1 follows the first residual, Norm2 follows the second.)

### 5.8 Functional Options Wiring

```go
// [REFERENCE] In pkg/nn/options.go.

// WithEncoderBlock attaches a single EncoderBlock[T] using the provided config.
func WithEncoderBlock[T utils.Float](cfg transformer.TransformerConfig[T]) Option[T] { /* ... */ }

// WithDecoderBlock attaches a single DecoderBlock[T].
func WithDecoderBlock[T utils.Float](cfg transformer.TransformerConfig[T]) Option[T] { /* ... */ }

// WithEncoderStack attaches a Stack[T] of N encoder blocks.
func WithEncoderStack[T utils.Float](cfg transformer.TransformerConfig[T], n int) Option[T] { /* ... */ }

// WithDecoderStack attaches a Stack[T] of N decoder blocks.
func WithDecoderStack[T utils.Float](cfg transformer.TransformerConfig[T], n int) Option[T] { /* ... */ }
```

All four write to a new `config.TransformerLayers []layer.Layer[T]` slice on `Config[T]`. compile() inserts the Transformer layers after any embedding layer and before the final classifier head.

### 5.9 Persistence Envelope Example

```json
{
  "Type": "transformer.EncoderBlock",
  "Config": {
    "SeqLen": 64,
    "Dmodel": 128,
    "NumHeads": 8,
    "Dff": 512,
    "PreNorm": true,
    "DropoutRate": 0.1,
    "Activation": "ReLU"
  },
  "Attn": { /* MultiHeadAttention JSON per ATT-9 */ },
  "Norm1": { /* LayerNorm JSON */ },
  "Norm2": { /* LayerNorm JSON */ },
  "FFN1": { /* Dense JSON */ },
  "FFN2": { /* Dense JSON */ }
}
```

Stack envelope wraps this:

```json
{
  "Type": "transformer.Stack",
  "Config": { /* same as block config */ },
  "Mode": "encoder",
  "Blocks": [ { /* EncoderBlock JSON */ }, { /* ... */ } ]
}
```

## 6. Implementation Notes

L2 realization should land in three sub-phases aligned with l1-transformer-block.md §6:

### Phase α — EncoderBlock (post-norm baseline) (4-5 atomic tasks)

- α.1: `pkg/layer/transformer/config.go` — `TransformerConfig[T]` struct + `Mode` enum + activation kind shim if needed.
- α.2: `pkg/layer/transformer/block.go` — `Block[T]` interface + `addInPlace[T]` helper + child-construction helpers.
- α.3: `pkg/layer/transformer/encoder.go` — `EncoderBlock[T]` struct + `NewEncoderBlock` + post-norm `Forward` + `Backward` + `Init` + JSON.
- α.4: `pkg/layer/transformer/encoder_test.go` — forward/backward finite-diff, JSON round-trip, post-norm composition correctness.
- α.5: `pkg/layer/transformer/residual_test.go` — TRANS-6 identity test (zero-weight init → output=input bit-exact).

### Phase β — Pre-norm + Dropout (2-3 atomic tasks)

- β.1: Add `PreNorm` branch to `encoder.go::Forward` and `Backward` per TRANS-3.
- β.2: Wire `Drop1` / `Drop2` reusing `*regularizer.Dropout[T]`; verify forward equivalence at `Cfg.DropoutRate == 0`.
- β.3: `pkg/layer/transformer/prenorm_test.go` — pre-norm vs post-norm equivalence on independent random init at Dropout=0; gradient flow sanity check at depth=12.

### Phase γ — DecoderBlock + Stack (3-4 atomic tasks)

- γ.1: `pkg/layer/transformer/decoder.go` — `DecoderBlock[T]` with `Causal: true` forwarded to MHA constructor.
- γ.2: `pkg/layer/transformer/stack.go` — `Stack[T]` + `NewStack` + Forward / Backward chain + `SetPaddingMask` forwarding.
- γ.3: `pkg/nn/options.go` — four `With*` options.
- γ.4: `pkg/layer/transformer/stack_test.go` — 6-block stack convergence on a synthetic copy-task; parameter count matches §4.4 BERT-Base-style math.

### Coverage targets

- `pkg/layer/transformer/`: ≥ 85% per C30 (composite layers tend to have lower coverage due to delegated logic; aggressive child-mocking is unnecessary — exercise real children).
- E17+ examples authored in a later examples phase exercise end-to-end (Embedding → Stack → Loss).

## 7. Drawbacks & Alternatives

- **Alternative: Single `Block[T]` public type with mode enum (Encoder / Decoder)** — accepted as a future refactor. v0.1 keeps the two types separate for code clarity. The internal `Block[T]` interface allows the Stack to be uniform without forcing the consolidation now.
- **Alternative: Slice-of-Layer[T] for children instead of typed fields** — rejected. Typed fields enable zero-cost child access in the hot path (no type assertion per Forward call). The wiring is fixed and small (5 children per block), so the rigidity of typed fields is the right trade-off.
- **Alternative: Expose intermediate activations (attention map, FFN hidden state) for analysis** — out of scope for v0.1. The block's `Forward` returns only the final output. Analysis access lives on the inner MHA (already exposed via `attention.MultiHeadAttention[T]` fields per ATT-7 cache). A future amendment may add an `IntrospectionLayer[T]` interface if research workloads demand it.
- **Alternative: Per-block heterogeneous config in Stack (different Dmodel / NumHeads per layer)** — out of scope. Mixed-config stacks are research-grade rather than production. Stack homogeneity simplifies validation and matches every production architecture (BERT, GPT-2, ViT). A future amendment may add a `HeterogeneousStack[T]` if needed.
- **Alternative: Fused Attention + LayerNorm kernel (à la FlashAttention)** — out of scope for L2. The L1 contract permits fusion but does not require it. Fusion lands as a future amendment to `l2-backend-gpu.md` that intercepts compatible block patterns and replaces them with a fused kernel.
- **Alternative: Activation enum as an interface (caller injects custom activation)** — rejected for v0.1. The existing `pkg/activation/` dispatcher uses a `Kind` enum per C27. Following the same convention keeps the FFN activation discoverable and serializable; the trade-off is that custom activations require a `pkg/activation/` PR rather than user-side wiring.
- **Alternative: Owner-block aggregates a single composite JSON (no per-child object)** — rejected per TRANS-9. Per-child delegated JSON keeps composite blocks decoupled from child schema evolution; adding a field to `MultiHeadAttention[T]` does not require updating block marshalling code.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[PARENT]` | `.design/main/specifications/l1-transformer-block.md` | TRANS-1..TRANS-10 invariants + TRANS-C1..TRANS-C8 constraints — normative source |
| `[ATTN-IMPL]` | `.design/main/specifications/l2-attention-impl.md` | `MultiHeadAttention[T]` reused; `MaskedLayer[T]` forwarded |
| `[NORM-IMPL]` | `.design/main/specifications/l2-normalization-impl.md` | `LayerNorm[T]` reused; BatchNorm forbidden per TRANS-C5 |
| `[LAYER-IF]` | `.design/main/specifications/l2-layer-types.md` | `layer.Layer[T]` interface; `Dense[T]` reused for FFN |
| `[ACT]` | `.design/main/specifications/l2-activation-functions.md` | FFN activation via existing dispatcher; default ReLU |
| `[REG-IMPL]` | `.design/main/specifications/l2-regularization-impl.md` | `Dropout[T]` reused at three positions per TRANS-C7 |
| `[EMBED-IMPL]` | `.design/main/specifications/l2-embedding-impl.md` | Upstream producer of `(SeqLen, Dmodel)` input |
| `[INIT]` | `.design/main/specifications/l2-init-impl.md` | Xavier for FFN Dense matrices |
| `[PERSIST-IMPL]` | `.design/main/specifications/l2-persistence-impl.md` | JSON envelope; child delegation per TRANS-9 |
| `[TRAIN-LOOP]` | `.design/main/specifications/l2-training-loop.md` | Backward routes upstream through children in reverse order |

<!-- Downstream agent instruction: when implementing this package, the three phases must run sequentially — α before β before γ. The PreNorm branch (β) shares state with the post-norm branch — implement them in a single Forward method with a top-level `if e.Cfg.PreNorm` rather than two separate functions; this keeps the buffer-reuse logic centralized. The TRANS-6 residual identity test (residual_test.go) is the single most important correctness check — if it fails, EVERY downstream behavior is suspect. Run this test first in every CI invocation. Forward equivalence between pre-norm and post-norm at Dropout=0 (prenorm_test.go) is a soft check — it requires identical init seeds for both code paths and matches the algebraic equivalence at zero residual scaling. -->

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-19 | Initial L2 spec authored via `/magic-spec` Blank Trigger (Creative Spark 1). Implements all 10 L1 invariants (TRANS-1..TRANS-10) with full Invariant Compliance table. Package layout `pkg/layer/transformer/` with 11 files (6 source + 5 tests). Typed child-ownership pattern (Attn / Norm1 / Norm2 / FFN1 / FFN2 / Drop1 / Drop2 as named struct fields) rather than `[]Layer[T]` for zero-cost hot-path access. Private `Block[T]` interface so `Stack[T]` holds encoder and decoder blocks uniformly. Persistence delegates to children per TRANS-9 — no field-level remarshalling. Four new `pkg/nn/` options. 3-phase implementation plan (α EncoderBlock post-norm baseline → β PreNorm + Dropout → γ DecoderBlock + Stack). Coverage target ≥ 85% per C30. Promoted Draft → Stable via Trust Mode (C9): MVC satisfied (Overview + §4 Invariant Compliance + §5 Detailed Design); no RULES.md conflicts; composes existing Stable primitives (Attention, LayerNorm, Dense, Dropout, Activation); layer constraint satisfied (Implements points to Stable L1 parent `l1-transformer-block.md`). |
