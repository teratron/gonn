# Embedding Layers — Go Implementation

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-embedding-layers.md

## Overview

Concrete Go realization of [l1-embedding-layers.md](l1-embedding-layers.md) — embedding layers delivered as the `pkg/layer/embedding/` package. Three exported types: `TokenEmbedding[T]` (vocabulary lookup table, `(V, Dmodel)` flat `[]T` matrix with sparse gradient), `PositionalEncoding[T]` (single type with `Mode` field switching between `Sinusoidal` and `Learnable` per `PositionalMode` enum — single struct mirrors the `MultiHeadAttention[T]` / `NumHeads` pattern from `l2-attention-impl.md`), and `EmbeddingStack[T]` (composes one TokenEmbedding + one PositionalEncoding via element-wise addition). All three satisfy `layer.Layer[T]` so the topology graph treats them uniformly with Dense / Conv / Recurrent / Attention.

Adds three `pkg/nn/` functional options (`WithTokenEmbedding`, `WithPositionalEncoding`, `WithEmbeddingStack`), one new error sentinel (`ErrVocabOutOfRange`) in `pkg/utils/errors.go`, and a sparse-gradient helper struct `sparseGrad[T]` in `pkg/layer/embedding/sparse.go`.

## Related Specifications

- [l1-embedding-layers.md](l1-embedding-layers.md) — Parent — EMB-1..EMB-10 invariants + EMB-C1..EMB-C8 constraints
- [l2-layer-types.md](l2-layer-types.md) — `Layer[T]` interface this package satisfies
- [l2-attention-impl.md](l2-attention-impl.md) — Reference for `pkg/layer/attention/` layout pattern; single-struct-with-mode pattern mirrored here for PositionalEncoding
- [l2-conv-layers-impl.md](l2-conv-layers-impl.md) — Reference for `pkg/layer/conv/` option-wiring style
- [l2-init-impl.md](l2-init-impl.md) — Xavier initialization helper for `W_E` and learnable `W_P`
- [l2-training-loop.md](l2-training-loop.md) — Backward dispatcher; embedding layers plug in at the standard layer-backward extension point with sparse-gradient routing
- [l2-persistence-impl.md](l2-persistence-impl.md) — JSON round-trip for `W_E` and learnable `W_P` per EMB-8; sinusoidal regenerates from formula
- [l2-errors-impl.md](l2-errors-impl.md) — `ErrVocabOutOfRange` sentinel added for EMB-7
- [l2-optimizer-impl.md](l2-optimizer-impl.md) — Optimizers consume sparse gradient via `ParamAccessor[T]` extension; SGD baseline applies updates only to touched rows

## 1. Motivation

L1 fixes the math (EMB-1 lookup, EMB-4 sinusoidal formula, EMB-6 additive composition) and the contracts (EMB-7 bounds, EMB-9 sparse backward). This spec fixes the Go specifics:

- **Package path**: `pkg/layer/embedding/` — peer to `pkg/layer/attention/`, `pkg/layer/conv/`, `pkg/layer/recurrent/`, `pkg/layer/norm/`.
- **Integer-ID input shape**: a new `Forward([]int) []T` overload alongside the existing `Forward([]T) []T`. The `layer.Layer[T]` interface is extended with an optional `IDLayer[T]` sub-interface — only embedding layers implement it; the topology compiler dispatches via a type assertion at `compile()`.
- **Sparse gradient representation**: `sparseGrad[T]` is a `(touched-rows bitset, dense [V][Dmodel]T accumulator)` pair. Accumulation is dense-write-with-bitset-set; optimizer Step iterates the bitset and applies updates only to set rows. Memory: O(V/8 + V × Dmodel) per layer, but per-step work is O(unique IDs × Dmodel) per EMB-9.
- **Compile-time interface assertion**: explicit `var _ layer.Layer[float64] = (*TokenEmbedding[float64])(nil)` etc per C26.
- **Sinusoidal table caching**: sinusoidal `PositionalEncoding[T]` precomputes the `(MaxSeqLen, Dmodel)` table once in `Init` and reuses it. `MarshalJSON` skips the table; `UnmarshalJSON` rebuilds it. Saves `MaxSeqLen × Dmodel × 8` bytes of JSON per layer.
- **Option wiring**: three new options propagate the embedding layer onto a new `config.EmbeddingLayer` field, consumed by `compile()` as the input-side layer ahead of any downstream sequence layer. The topology validator confirms the downstream layer's expected `Dmodel` matches.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| EMB-1 (TokenEmbedding shape) | `Forward([]int) []T` accepts `(SeqLen,)` of length `SeqLen`; returns `[]T` of length `SeqLen × Dmodel` (row-major flat); lookup is `copy(out[i*Dmodel : (i+1)*Dmodel], W_E[IDs[i]*Dmodel : (IDs[i]+1)*Dmodel])` per row |
| EMB-2 (TokenEmbedding parameters) | `W_E []T` flat row-major `[V × Dmodel]`; no bias field; `V` and `Dmodel` stored as `int` fields, persisted in JSON config |
| EMB-3 (PositionalEncoding shape) | `Forward(seqLen int) []T` returns `P[0 : seqLen*Dmodel]` — slice of the `(MaxSeqLen, Dmodel)` table truncated to the actual input length; reuses the same backing array — no copy |
| EMB-4 (Sinusoidal formula) | `buildSinusoidalTable(maxSeqLen, dmodel int) []T` in `table.go`: loops `pos × k`, computes `freq := T(1) / math.Pow(10000, T(2*k)/T(dmodel))`, stores `sin(pos*freq)` at `[pos*Dmodel + 2k]` and `cos(pos*freq)` at `[pos*Dmodel + 2k+1]`; odd-`Dmodel` final dim gets `sin` only; test `TestSinusoidalFormula` verifies row-0 is `[0, 1, 0, 1, ...]` and exact match against reference values for `(pos=1, Dmodel=4)` |
| EMB-5 (Learnable positional parameters) | When `Mode == Learnable`: `W_P []T` flat `[MaxSeqLen × Dmodel]`; `Init(rng)` calls `utils.Xavier[T](rng, Dmodel)` to scale by `sqrt(1/Dmodel)`; touched-rows bitset reused from `sparseGrad` for per-batch dense subset gradient |
| EMB-6 (EmbeddingStack composition) | `EmbeddingStack[T].Forward(ids []int) []T` calls `tok.Forward(ids)` then `for i, v := range pos.Forward(len(ids)) { out[i] += v }` — single in-place loop fused with the lookup write |
| EMB-7 (Vocabulary bounds) | `TokenEmbedding[T].Forward` validates `ids[i] < 0 \|\| ids[i] >= V` per element; on violation returns `fmt.Errorf("embedding lookup at position %d: id %d out of vocabulary [0, %d): %w", i, ids[i], V, ErrVocabOutOfRange)`; new sentinel `ErrVocabOutOfRange` added to `pkg/utils/errors.go` wrapping `ErrUserConfig` per C32 |
| EMB-8 (Persistence) | `TokenEmbedding[T].MarshalJSON` serialises `{Type:"token", V, Dmodel, W_E}`; `PositionalEncoding[T].MarshalJSON` with `Mode == Sinusoidal` writes `{Type:"positional", Mode:"sinusoidal", MaxSeqLen, Dmodel}` (no table); with `Mode == Learnable` writes `{Type:"positional", Mode:"learnable", MaxSeqLen, Dmodel, W_P}`; `UnmarshalJSON` branches on `Mode` and calls `buildSinusoidalTable` for sinusoidal load |
| EMB-9 (Sparse gradient — TokenEmbedding) | `Backward(dOut []T) (dIn []T, err error)`: `dIn` is `nil` (input is integer IDs — not differentiable); accumulation iterates the cached `ids[]` from Forward: `sparseGrad.add(ids[i], dOut[i*Dmodel:(i+1)*Dmodel])` per row; optimizer Step reads `sparseGrad.iter(func(row int, grad []T))` and applies updates only to set rows; bitset cleared after each Step; learnable PositionalEncoding uses identical sparseGrad scoped to `[0, SeqLen)` rows |
| EMB-10 (Layer interface) | Compile-time assertions in each file: `var _ layer.Layer[float64] = (*TokenEmbedding[float64])(nil)` (in `token.go`), same for `PositionalEncoding[float64]` (in `positional.go`) and `EmbeddingStack[float64]` (in `stack.go`); new `IDLayer[T]` sub-interface in `pkg/layer/core.go` with `ForwardIDs([]int) ([]T, error)` — only TokenEmbedding and EmbeddingStack implement it; compile() dispatches via type assertion |

## 5. Detailed Design

### 5.1 Package Layout

```text
pkg/layer/embedding/
├── doc.go              // package documentation + AI-Meta block
├── table.go            // buildSinusoidalTable; xavier init helper passthrough; sparse-grad bitset utility
├── sparse.go           // sparseGrad[T] — touched-rows bitset + dense accumulator + iter
├── token.go            // TokenEmbedding[T] — lookup, Backward (sparse), JSON, IDLayer impl
├── positional.go       // PositionalEncoding[T] + PositionalMode enum (Sinusoidal | Learnable); single struct
├── stack.go            // EmbeddingStack[T] — composes TokenEmbedding + PositionalEncoding
├── token_test.go       // lookup correctness; bounds error (EMB-7); JSON round-trip; sparse backward; finite-diff
├── positional_test.go  // sinusoidal formula (EMB-4 row-0 + reference values); learnable JSON round-trip;
│                       //   sinusoidal-no-table JSON shape; backward (learnable only)
├── stack_test.go       // additive composition; backward routing to both children; pattern A end-to-end
└── sparse_test.go      // sparseGrad add/iter/reset cycle; bitset correctness on dense and sparse patterns
```

### 5.2 TokenEmbedding Struct

```go
// [REFERENCE] In pkg/layer/embedding/token.go.
package embedding

import (
    "github.com/teratron/gonn/pkg/layer"
    "github.com/teratron/gonn/pkg/utils"
)

type TokenEmbedding[T utils.Float] struct {
    V      int  // vocabulary size
    Dmodel int  // output feature dimension

    // W_E [V × Dmodel] row-major flat. Row IDs[i] is the embedding for token IDs[i].
    W_E []T

    // Forward cache — populated by Forward, consumed by Backward.
    ids   []int       // last input ID sequence
    grads *sparseGrad[T]
}

var _ layer.Layer[float32] = (*TokenEmbedding[float32])(nil)
var _ layer.Layer[float64] = (*TokenEmbedding[float64])(nil)
var _ layer.IDLayer[float32] = (*TokenEmbedding[float32])(nil)
var _ layer.IDLayer[float64] = (*TokenEmbedding[float64])(nil)

// ForwardIDs implements IDLayer[T] — embedding-specific entry that takes integer IDs directly.
//
// AI-Meta:
// - Purpose: Look up token IDs to their dense embedding vectors.
// - Usage: ids = caller-supplied integer slice (e.g., from a tokenizer).
// - Lifecycle: Forward-only; gradient accumulation happens in Backward.
// - Concurrency: SingleGoroutine — caller serializes Forward and Backward calls.
// - Errors: ErrVocabOutOfRange when any ID is outside [0, V).
// - Stability: Experimental.
func (e *TokenEmbedding[T]) ForwardIDs(ids []int) ([]T, error) {
    // ... bounds check + row copy + cache ids for Backward
}
```

### 5.3 PositionalEncoding Single-Struct Pattern

```go
// [REFERENCE] In pkg/layer/embedding/positional.go.
package embedding

type PositionalMode uint8

const (
    Sinusoidal PositionalMode = iota
    Learnable
)

func (m PositionalMode) String() string {
    switch m {
    case Sinusoidal:
        return "sinusoidal"
    case Learnable:
        return "learnable"
    }
    return "unknown"
}

type PositionalEncoding[T utils.Float] struct {
    Mode       PositionalMode
    MaxSeqLen  int
    Dmodel     int

    // Table — [MaxSeqLen × Dmodel] row-major flat.
    //   Sinusoidal: regenerated in Init / UnmarshalJSON, never persisted.
    //   Learnable:  this is W_P, persisted in JSON.
    Table []T

    // Forward cache — only used for learnable backward.
    lastSeqLen int
    grads      *sparseGrad[T]  // touched only for Mode == Learnable
}

var _ layer.Layer[float32] = (*PositionalEncoding[float32])(nil)
var _ layer.Layer[float64] = (*PositionalEncoding[float64])(nil)
```

This single-struct + mode-enum pattern mirrors `l2-attention-impl.md` §5.2 (one `MultiHeadAttention[T]` covering single-head and multi-head cases via `NumHeads`). Single canonical struct avoids interface dispatch overhead in the hot path and simplifies persistence.

### 5.4 Sparse Gradient Helper

```go
// [REFERENCE] In pkg/layer/embedding/sparse.go.
package embedding

// sparseGrad accumulates gradient rows into a (V, Dmodel) buffer and tracks
// which rows have been touched in the current batch via a bitset.
type sparseGrad[T utils.Float] struct {
    Rows    int        // V (or MaxSeqLen for learnable positional)
    Dmodel  int
    Buf     []T        // [Rows × Dmodel] dense buffer
    Touched []uint64   // bitset of length ceil(Rows / 64)
    NumSet  int        // population count, for quick empty check
}

// add accumulates row gradient into Buf[row] and marks row as touched.
//
// AI-Meta:
// - Purpose: Accumulate per-row gradient for sparse embedding update.
// - Usage: Called once per token position during Backward.
// - Concurrency: SingleGoroutine — embedding layer owns this state.
// - Constraints: row must be in [0, Rows).
// - Stability: Internal.
func (s *sparseGrad[T]) add(row int, grad []T) { /* ... */ }

// iter walks each touched row and invokes fn(row, grad).
//
// AI-Meta:
// - Purpose: Drive the optimizer Step over only the rows touched in the batch.
// - Usage: Called from the optimizer's ParamAccessor extension.
// - Concurrency: SingleGoroutine.
// - Stability: Internal.
func (s *sparseGrad[T]) iter(fn func(row int, grad []T)) { /* ... */ }

// reset clears the touched bitset (keeps Buf — rows are overwritten on next add).
func (s *sparseGrad[T]) reset() { /* ... */ }
```

### 5.5 IDLayer Sub-Interface

A new optional sub-interface declared in `pkg/layer/core.go`:

```go
// [REFERENCE] In pkg/layer/core.go.
package layer

// IDLayer is the optional sub-interface implemented by layers that accept
// integer ID input rather than a continuous-valued tensor. Currently only
// TokenEmbedding and EmbeddingStack satisfy it. The topology compiler
// dispatches via type assertion at compile() time.
//
// AI-Meta:
// - Purpose: Allow embedding layers to consume integer ID sequences directly.
// - Usage: compile() checks if the input layer implements IDLayer to route data flow.
// - Stability: Experimental.
type IDLayer[T utils.Float] interface {
    Layer[T]
    ForwardIDs(ids []int) ([]T, error)
}
```

The compile() pass detects when the first layer is an `IDLayer[T]` and accordingly types the training loop input as `[][]int` (sequences of IDs) instead of `[][]T` (continuous tensors). Subsequent layers consume the float output of the embedding layer normally.

### 5.6 Functional Options Wiring

```go
// [REFERENCE] In pkg/nn/options.go.

// WithTokenEmbedding attaches a TokenEmbedding[T] as the input-side layer.
// V is vocabulary size, dmodel is the output feature dimension forwarded to
// downstream sequence layers.
func WithTokenEmbedding[T utils.Float](v, dmodel int) Option[T] { /* ... */ }

// WithPositionalEncoding attaches a PositionalEncoding[T] using the given mode.
// maxSeqLen caps the sequence length the layer can encode.
func WithPositionalEncoding[T utils.Float](mode embedding.PositionalMode, maxSeqLen, dmodel int) Option[T] { /* ... */ }

// WithEmbeddingStack attaches an EmbeddingStack[T] — the canonical
// (TokenEmbedding + PositionalEncoding) composition for NLP pipelines.
func WithEmbeddingStack[T utils.Float](v, dmodel, maxSeqLen int, mode embedding.PositionalMode) Option[T] { /* ... */ }
```

All three options write to a new `config.EmbeddingLayer layer.Layer[T]` field. compile() asserts at most one is set and that the downstream layer's input feature dimension equals `Dmodel`.

### 5.7 Sparse Gradient Optimizer Path

Existing optimizers (`SGD`, `Adam`, `RMSProp`) iterate dense parameter slices via `ParamAccessor[T]`. For embedding layers, the layer exposes its `sparseGrad` via a new optional `SparseParamAccessor[T]` extension:

```go
// [REFERENCE] In pkg/optimizer/sparse.go.

// SparseParamAccessor[T] is the optional optimizer extension implemented by
// layers with sparse gradient (TokenEmbedding, learnable PositionalEncoding).
// Optimizers detect this via type assertion and dispatch to the sparse path
// to avoid dense iteration over an unchanged matrix.
//
// AI-Meta:
// - Purpose: Allow optimizers to apply updates only to gradient rows that were touched in the batch.
// - Usage: Optimizer's Step method type-asserts the layer to this interface.
// - Stability: Experimental.
type SparseParamAccessor[T utils.Float] interface {
    // SparseRows iterates only the rows that received gradient this step.
    SparseRows(fn func(row int, param, grad []T))
}
```

The L2 implementation MUST update SGD as a baseline; Adam and RMSProp also need per-row moment slots — initial v0.1 may apply moments per touched row only, accepting the simplification that untouched rows decay moments at the next call rather than every step. Documented as a known approximation in the test file `sparse_test.go`.

### 5.8 Shape & Dispatch Compatibility

| Downstream Layer | Embedding Output | Notes |
| :--- | :--- | :--- |
| `MultiHeadAttention[T]` | `(SeqLen, Dmodel)` flat `[]T` of length `SeqLen × Dmodel` | Direct — Attention's input shape contract matches |
| `LSTM[T]` / `SimpleRNN[T]` / `GRU[T]` | same | Same flat layout; recurrent layers iterate per-position |
| `Conv1D[T]` | same | Conv1D treats Dmodel as channel dim, SeqLen as spatial |
| `Dense[T]` | requires Flatten upstream | Embedding output is 2-D; Dense expects 1-D — user inserts `Flatten[T]` between |

## 6. Implementation Notes

Three sub-phases for the L2 implementation, aligned with l1-embedding-layers §6:

### Phase α — TokenEmbedding baseline (3-4 atomic tasks)

- α.1: `pkg/layer/embedding/sparse.go` — `sparseGrad[T]` with `add` / `iter` / `reset`; full bitset test coverage.
- α.2: `pkg/layer/embedding/token.go` — `TokenEmbedding[T]` struct + `ForwardIDs` + `Backward` (sparse) + `Init` (Xavier on `Dmodel`) + JSON round-trip.
- α.3: `pkg/layer/core.go` — `IDLayer[T]` sub-interface declaration.
- α.4: `pkg/utils/errors.go` — `ErrVocabOutOfRange` sentinel wrapping `ErrUserConfig`.

### Phase β — PositionalEncoding (3 atomic tasks)

- β.1: `pkg/layer/embedding/table.go` — `buildSinusoidalTable[T]` with formula correctness test.
- β.2: `pkg/layer/embedding/positional.go` — single `PositionalEncoding[T]` struct + `PositionalMode` enum + `Forward(seqLen int)` + `Backward` (learnable only) + JSON branching on `Mode`.
- β.3: Test coverage — sinusoidal row-0 invariant, reference values, learnable Xavier init, no-table sinusoidal JSON shape.

### Phase γ — EmbeddingStack + nn integration (3-4 atomic tasks)

- γ.1: `pkg/layer/embedding/stack.go` — `EmbeddingStack[T]` composition + IDLayer impl + JSON owns both children.
- γ.2: `pkg/nn/options.go` — three `With*` options + `config.EmbeddingLayer` field.
- γ.3: `pkg/nn/compile.go` — embedding-as-input-layer dispatch; topology validator checks Dmodel match with downstream layer.
- γ.4: `pkg/optimizer/sparse.go` — `SparseParamAccessor[T]` extension + SGD baseline integration; Adam/RMSProp approximation deferred to future phase.

### Coverage targets

- `pkg/layer/embedding/`: ≥ 85% per C30.
- New optimizer sparse path: ≥ 80%.
- E17 example (`examples/text_classification/`) for end-to-end smoke validation in a later examples phase.

## 7. Drawbacks & Alternatives

- **Alternative: Bake sparse gradient into the existing `ParamAccessor[T]` rather than introducing `SparseParamAccessor[T]`** — rejected. Existing optimizers would pay a per-step type check even for layers without sparse gradient. The optional sub-interface keeps the fast path uncluttered, matching the precedent set by `LearningRateSetter[T]` in `l2-lr-scheduling-impl.md`.
- **Alternative: Two separate types `SinusoidalPositionalEncoding[T]` and `LearnablePositionalEncoding[T]`** — rejected. Increases public surface for negligible gain. The single-struct + mode-enum pattern follows the `MultiHeadAttention[T]` / `NumHeads` precedent and keeps the option / persistence surface uniform. Performance impact is one extra integer compare per Forward; negligible.
- **Alternative: Implement `Forward([]T)` on embedding by reinterpreting the float input as IDs** — rejected. Type-unsafe (silently misinterprets continuous inputs); incompatible with C25 generic constraint. The IDLayer sub-interface is the type-correct route.
- **Alternative: Embedding output as `[][]T` 2-D slice** — rejected. The library's convention (set by attention, recurrent, conv) is flat `[]T` row-major. Maintaining one convention reduces conversion at layer boundaries.
- **Alternative: Hashed embedding tables (Hash Trick)** — out of scope. Hashed embeddings collapse vocabulary with a hash function, reducing parameters at the cost of collision noise. A future amendment for very-large-vocabulary deployment scenarios.
- **Alternative: `SparseTensor[T]` first-class type in `pkg/utils/`** — rejected for v0.1. Premature abstraction — no other layer needs sparse representation. The `sparseGrad[T]` helper is intentionally embedding-package-private. If a third spec needs sparse tensors, refactor at that point.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[PARENT]` | `.design/main/specifications/l1-embedding-layers.md` | EMB-1..EMB-10 invariants + EMB-C1..EMB-C8 constraints — normative source |
| `[LAYER-IF]` | `.design/main/specifications/l2-layer-types.md` | `layer.Layer[T]` interface this package satisfies |
| `[ATTN-PATTERN]` | `.design/main/specifications/l2-attention-impl.md` | Reference layout for `pkg/layer/attention/`; single-struct-with-mode pattern (NumHeads) mirrored for PositionalEncoding |
| `[CONV-PATTERN]` | `.design/main/specifications/l2-conv-layers-impl.md` | Reference for option-wiring style and compile() dispatch in `pkg/nn/` |
| `[INIT]` | `.design/main/specifications/l2-init-impl.md` | `utils.Xavier[T]` for `W_E` and learnable `W_P` |
| `[ERRORS]` | `.design/main/specifications/l2-errors-impl.md` | Adds `ErrVocabOutOfRange` sentinel per EMB-7 |
| `[PERSIST]` | `.design/main/specifications/l2-persistence-impl.md` | JSON envelope; sinusoidal skips table per EMB-C5 |
| `[OPTIMIZER]` | `.design/main/specifications/l2-optimizer-impl.md` | Adds optional `SparseParamAccessor[T]` extension; SGD baseline updated |
| `[TRAIN-LOOP]` | `.design/main/specifications/l2-training-loop.md` | Backward dispatcher routes IDLayer differently — `dIn` is nil for ID-based inputs |

<!-- Downstream agent instruction: when implementing this package, follow the three-phase order strictly. α before β before γ — Phase γ depends on both prior phases compiling. The sparseGrad helper (α.1) is the foundation — get its tests green before moving to TokenEmbedding (α.2). For Adam / RMSProp moments under sparse update, the v0.1 approximation (per-touched-row moment decay) is acceptable; document it in sparse_test.go as a known divergence from dense-equivalent behavior. -->

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-19 | Initial L2 spec authored via `/magic-spec` Blank Trigger (Creative Spark 2). Implements all 10 L1 invariants (EMB-1..EMB-10) with full Invariant Compliance table. Package layout `pkg/layer/embedding/` with 9 files (3 source + 1 enum + 5 tests); single `PositionalEncoding[T]` struct with `PositionalMode` enum mirrors `MultiHeadAttention[T]` / `NumHeads` precedent. New `IDLayer[T]` sub-interface in `pkg/layer/core.go` for integer-ID input. New `SparseParamAccessor[T]` optimizer extension for sparse gradient update (SGD baseline; Adam/RMSProp approximation noted). New `ErrVocabOutOfRange` sentinel. 3-phase implementation plan (α TokenEmbedding + sparse + IDLayer → β PositionalEncoding both modes → γ EmbeddingStack + nn options + compile() dispatch). Coverage target ≥ 85% per C30. Promoted Draft → Stable via Trust Mode (C9): MVC satisfied (Overview + §4 Invariant Compliance + §5 Detailed Design); no RULES.md conflicts; layer constraint satisfied (Implements points to Stable L1 parent `l1-embedding-layers.md`). |
