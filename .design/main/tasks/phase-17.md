---
phase: 17
name: "NLP Foundation — Attention Implementation + Embedding Layers"
status: Todo
subsystem: "pkg/layer/attention/ (new — MultiHeadAttention[T] + cell.go softmax helpers + MaskedLayer[T]); pkg/layer/embedding/ (new — TokenEmbedding[T] + PositionalEncoding[T] + EmbeddingStack[T] + sparseGrad[T]); pkg/layer/core.go (new IDLayer[T] sub-interface); pkg/optimizer/ (new SparseParamAccessor[T] extension + SGD sparse path); pkg/utils/errors.go (ErrVocabOutOfRange sentinel); pkg/nn/ (six new options + compile() dispatch for attention prefix + embedding-as-input-layer)"
requires:
  - "Phase 16 ✓ (v0.14.0 RC ready — recurrent completion + GPU backward + AI-Meta rollout)"
  - "l1-attention Stable v0.1.0 ✓ (ATT-1..10 invariants)"
  - "l2-attention-impl Stable v0.1.0 ✓ (§6 5-phase plan α-ε)"
  - "l1-embedding-layers Stable v0.1.0 ✓ (EMB-1..10 invariants — authored 2026-05-19 same-session)"
  - "l2-embedding-impl Stable v0.1.0 ✓ (§6 3-phase plan α-β-γ — authored 2026-05-19 same-session)"
  - "l1-weight-initialization Stable v1.0.0 ✓ (Xavier helper reused by both tracks)"
  - "l1-network-persistence Stable v1.0.0 ✓ (JSON round-trip contract)"
  - "l2-init-impl Stable v1.0.0 ✓ (utils.Xavier[T] reused)"
  - "Phase 15 patterns_established: build-tag isolation, TestAIMetaCompliance hook template; Phase 14 patterns_established: layer prefix integration in compile() (Conv2D template)"
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 17 Tasks — NLP Foundation: Attention Implementation + Embedding Layers

**Phase:** 17
**Status:** Todo
**Strategic Goal:** Land the two independent NLP-stack primitives whose L1+L2 specs are already
Stable, in two fully parallel tracks within a single phase.

(A) **Attention Implementation** — realize `l2-attention-impl.md §6 α-ε` as `pkg/layer/attention/`:
single canonical `MultiHeadAttention[T]` struct covering all three L1 conceptual variants
(Attention / SelfAttention / MultiHeadAttention) via `NumHeads` field; head-major flat layout
for per-head Q/K/V index arithmetic (zero-copy splitHeads/joinHeads); softmax row-wise +
masked + backward helpers in `cell.go`; `MaskedLayer[T]` interface for runtime padding mask;
ATT-7 four-path backward (output projection + V path + softmax-backward + Q/K paths) verified
via finite-difference within 1e-4; ATT-3 scaling-factor entropy bound test; three new `pkg/nn/`
options (`WithAttention` / `WithMultiHeadAttention` / `WithCausalAttention`); attention prefix
slot in `compile()` after Recurrent layers and before Dense stack.

(B) **Embedding Implementation** — realize `l2-embedding-impl.md §6 α-β-γ` as `pkg/layer/embedding/`:
`TokenEmbedding[T]` with vocabulary lookup table `W_E [V × Dmodel]` row-major flat and EMB-9
sparse-gradient update (`sparseGrad[T]` with touched-rows bitset); single `PositionalEncoding[T]`
struct with `PositionalMode` enum (Sinusoidal / Learnable) — Sinusoidal regenerates table from
formula on load (zero persisted parameters); `EmbeddingStack[T]` additive composition
(TokenEmbedding + PositionalEncoding); new `IDLayer[T]` sub-interface on `pkg/layer/core.go`
for integer-ID input dispatch; new `SparseParamAccessor[T]` optimizer extension with SGD
baseline (Adam / RMSProp moments under sparse update deferred to Phase 18+); EMB-7 vocabulary
bounds enforcement with new `ErrVocabOutOfRange` sentinel; three new `pkg/nn/` options
(`WithTokenEmbedding` / `WithPositionalEncoding` / `WithEmbeddingStack`); embedding-as-input-layer
dispatch in `compile()` (replaces continuous input shape with integer-ID shape).

Phase 18 (Transformer Block implementation) is **scoping-gated** on this phase's closeout
gate T-17Z01 — the Transformer block reuses `MultiHeadAttention[T]` as its inner primitive
and consumes `EmbeddingStack[T]` output. Quantization L2 implementation (l1-quantization.md
§6) deferred to Phase 19+ per L1 spec author note. Target release: **v0.15.0**.

## Atomic Checklist

### Track A — Attention Implementation (parallel-safe: A01 ∥ A02 → A03 → A04; A05 last)

- [ ] [T-17A01] `pkg/layer/attention/cell.go` + `pkg/layer/attention/doc.go` — shared softmax helpers (`softmaxRowwise`, `softmaxRowwiseWithMask`, `softmaxBackwardRowwise`, `softmaxBackwardRowwiseWithMask`), scaling-factor precompute, head-major reshape index arithmetic helpers (`splitHeads`, `joinHeads`)
- [ ] [T-17A02] `pkg/layer/attention/multihead.go` — `MultiHeadAttention[T]` struct: `SeqLen`, `Dmodel`, `NumHeads`, `Dk` (cached), `Causal bool`, `Wq/Wk/Wv/Wo` + `Bq/Bk/Bv/Bo`, forward cache (`Qcache`, `Kcache`, `Vcache`, `Acache`); `Forward` (input projections → multi-head reshape view → scaled scores → optional causal mask → softmax → V multiply → joinHeads → output projection); `Init(rng)` calls `utils.Xavier[T](rng, Dmodel)` per ATT-8
- [ ] [T-17A03] `pkg/layer/attention/multihead.go` — `Backward` implementing ATT-7 four-path decomposition: (1) upstream → `Wo` + `Bo` + per-head `dA`; (2) `dA = dOut · Vᵀ`, `dV = Aᵀ · dOut`; (3) softmax-backward `dScores[i, :] = (dA[i, :] - <dA[i, :], A[i, :]>) ⊙ A[i, :]`; (4) `dQ = dScores · K / sqrt(Dk)`, `dK = dScoresᵀ · Q / sqrt(Dk)`; gradients to `Wq/Wk/Wv` via input projection backward; input gradient `∂L/∂X` accumulates from three input-projection paths
- [ ] [T-17A04] `pkg/layer/attention/masked_layer.go` — `MaskedLayer[T]` interface declaration with `SetPaddingMask([]bool)` method; `MultiHeadAttention[T].SetPaddingMask` implementation storing mask on layer state; per-call mask reset after consumption (per-call semantics); `applyCausalMask(scores, seqLen)` + `applyPaddingMask(scores, m)` helpers in `cell.go`; both masks compose additively before softmax
- [ ] [T-17A05] `pkg/layer/attention/multihead.go` JSON + `pkg/nn/options.go` + `pkg/nn/compile.go` — `MarshalJSON`/`UnmarshalJSON` per ATT-9 (serialize `{Type, SeqLen, Dmodel, NumHeads, Causal, Wq/Wk/Wv/Wo, Bq/Bk/Bv/Bo}`; padding mask NOT serialized); three pkg/nn options (`WithAttention[T]`, `WithMultiHeadAttention[T]`, `WithCausalAttention[T]`); `AttentionLayers []layer.Layer[T]` field on config; compile() slot — attention prefix inserted after Recurrent layers and before Dense stack; topology validator confirms `(SeqLen × Dmodel)` shape match with downstream layer

### Track B — Embedding Implementation (parallel-safe: B01 ∥ B02 → B03 → B04 → B05; B06 sequenced after A05)

- [ ] [T-17B01] `pkg/utils/errors.go` (ErrVocabOutOfRange sentinel wrapping `ErrUserConfig` per C32) + `pkg/layer/core.go` (`IDLayer[T]` sub-interface declaration with `ForwardIDs(ids []int) ([]T, error)` method — extends `Layer[T]`)
- [ ] [T-17B02] `pkg/layer/embedding/sparse.go` + `pkg/layer/embedding/doc.go` — `sparseGrad[T]` struct (`Rows`, `Dmodel`, `Buf []T`, `Touched []uint64` bitset, `NumSet int`); `add(row int, grad []T)` accumulates row gradient + marks bitset; `iter(fn func(row int, grad []T))` walks set rows only; `reset()` clears bitset (keeps `Buf` — rows overwritten on next add); package doc + AI-Meta block on each exported helper
- [ ] [T-17B03] `pkg/layer/embedding/token.go` — `TokenEmbedding[T]` struct: `V`, `Dmodel`, `W_E []T` (flat `[V × Dmodel]` row-major), `ids []int` (forward cache), `grads *sparseGrad[T]`; `ForwardIDs([]int) ([]T, error)` validates bounds per EMB-7 + copies rows from `W_E` to output flat `[]T` of length `SeqLen × Dmodel`; `Backward` accumulates upstream gradient into `grads` via `add(ids[i], dOut[i*Dmodel:(i+1)*Dmodel])` per row (sparse, EMB-9); `Init(rng)` calls `utils.Xavier[T](rng, Dmodel)`; `MarshalJSON`/`UnmarshalJSON` serialize `{Type:"token", V, Dmodel, W_E}`
- [ ] [T-17B04] `pkg/layer/embedding/table.go` + `pkg/layer/embedding/positional.go` — `buildSinusoidalTable[T](maxSeqLen, dmodel int) []T` per EMB-4 formula (sin at even indices, cos at odd; odd-Dmodel final dim sin-only); `PositionalEncoding[T]` single struct: `Mode PositionalMode` (enum `Sinusoidal`/`Learnable`), `MaxSeqLen`, `Dmodel`, `Table []T` (sinusoidal: regenerated on Init/UnmarshalJSON; learnable: this is `W_P`, persisted), `lastSeqLen` cache, `grads *sparseGrad[T]` for learnable; `Forward(seqLen int) []T` returns `Table[0 : seqLen*Dmodel]` slice (zero-copy); `Backward` (no-op for sinusoidal; sparse for learnable, only `[0, seqLen)` rows); JSON branches on Mode — sinusoidal writes `{Type:"positional", Mode:"sinusoidal", MaxSeqLen, Dmodel}` (no table), learnable writes `{Type:"positional", Mode:"learnable", MaxSeqLen, Dmodel, W_P}`
- [ ] [T-17B05] `pkg/layer/embedding/stack.go` — `EmbeddingStack[T]` struct owning one `*TokenEmbedding[T]` + one `*PositionalEncoding[T]`; `ForwardIDs([]int) ([]T, error)` calls `tok.ForwardIDs(ids)` then sums in `pos.Forward(len(ids))` element-wise via single in-place loop; `Backward` routes upstream gradient to both children (sparse to TokenEmbedding via stored ids, dense-but-position-scoped to learnable PositionalEncoding, no-op for sinusoidal); IDLayer + Layer compile-time assertions; JSON envelope owns both children
- [ ] [T-17B06] `pkg/nn/options.go` + `pkg/nn/compile.go` + `pkg/optimizer/sparse.go` — three pkg/nn options (`WithTokenEmbedding[T]`, `WithPositionalEncoding[T]`, `WithEmbeddingStack[T]`) writing to new `EmbeddingLayer layer.Layer[T]` field on config; compile() asserts at most one is set and downstream layer's input feature dim equals `Dmodel`; embedding-as-input-layer dispatch — type-assert input layer to `layer.IDLayer[T]`, route training data as `[][]int` (sequences of IDs) instead of `[][]T`; new `SparseParamAccessor[T]` optimizer extension with `SparseRows(fn func(row int, param, grad []T))` method; SGD baseline updated to dispatch via type assertion (Adam/RMSProp sparse moments documented as known approximation in `sparse_test.go`, full sparse-moment support deferred to Phase 18+)

### Validation

- [ ] [T-17T01] Validation Track A — attention forward correctness (4-head, 8-dim, 16-seqlen XOR fixture); ATT-7 finite-difference gradient check (max_abs_err < 1e-4 for T=float64); ATT-3 scaling-factor test asserts softmax-output entropy `> 0.5 * ln(SeqLen)` for `Dk = 64` (catches missing-scale bugs); causal mask correctness vs unmasked (post-softmax weight zero for `j > i`); padding mask + causal mask compose additively; JSON round-trip bit-exact; `pkg/layer/attention/` coverage ≥85%
- [ ] [T-17T02] Validation Track B — TokenEmbedding lookup correctness (V=10, Dmodel=4 fixture, hand-verified rows); EMB-7 bounds error message format (offending position + id + V reported); sinusoidal formula row-0 = `[0, 1, 0, 1, ...]` exact + reference value match for `(pos=1, Dmodel=4)`; sparse gradient finite-difference (V=20, Dmodel=8, batch of repeated IDs — verify per-row gradient accumulation); learnable PositionalEncoding gradient over `[0, SeqLen)` rows only; sinusoidal-no-table JSON envelope shape; EmbeddingStack additive composition (lookup row + position row identity check); `pkg/layer/embedding/` coverage ≥85%; new optimizer sparse path coverage ≥80%

### Gate

- [ ] [T-17Z01] Phase 17 release gate — `go build ./...` clean (default tags, no cgo); all new packages individually ≥80% coverage; `pkg/layer/attention/` ≥85%; `pkg/layer/embedding/` ≥85%; new optimizer sparse path ≥80%; `pkg/nn/` coverage maintained ≥75% (options + compile() growth offset); CHANGELOG.md v0.15.0 entry written; `v0.15.0` tag prepared (`git tag -a` left to user per finalization protocol); Phase 18 (Transformer Block) scoping condition met → `provides` field documents `MultiHeadAttention[T]` + `EmbeddingStack[T]` availability for downstream consumption

## Detailed Tracking

### [T-17A01] `pkg/layer/attention/cell.go` + `doc.go`

- **Spec:** `l2-attention-impl.md` §5.5 + ATT-3 (scaling) + ATT-4 (multi-head reshape)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestSoftmaxRowwise -count=1 ./pkg/layer/attention/` PASS; row-stochasticity of `softmaxRowwiseWithMask` under various mask patterns; numerical stability test at extreme scores (-1e6) — no NaN, no Inf; coverage line for `cell.go` ≥85%
- **Handoff:** A02 + A03 import softmax helpers; A04 uses mask helpers.
- **Notes:** Mirror layout of `pkg/layer/recurrent/cell.go` (Phase 15 precedent). Per-row max-subtract before exp for numerical stability. `splitHeads` and `joinHeads` are pure index arithmetic — no allocation, no copy. Helpers must be parameterized on `utils.Float` constraint per C25.

### [T-17A02] `pkg/layer/attention/multihead.go` (struct + Forward + Init)

- **Spec:** `l2-attention-impl.md` §5.2 + ATT-1 / ATT-2 / ATT-3 / ATT-4 / ATT-8
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestAttentionForward -count=1 ./pkg/layer/attention/` PASS; shape `(SeqLen × Dmodel) → (SeqLen × Dmodel)` preserved per ATT-1; `TestAttentionInitStats` asserts each projection matrix Frobenius norm within `[0.5·sqrt(Dmodel), 1.5·sqrt(Dmodel)]` per ATT-8; `var _ layer.Layer[float64] = (*MultiHeadAttention[float64])(nil)` compile-time assertion compiles
- **Handoff:** A03 implements Backward using forward cache; A04 wires mask via SetPaddingMask.
- **Notes:** Forward cache layout: `Qcache`/`Kcache`/`Vcache` flat `[NumHeads × SeqLen × Dk]`; `Acache` flat `[NumHeads × SeqLen × SeqLen]`. ATT-C3 enforced — constructor returns error if `Dmodel % NumHeads != 0`. Use existing `utils.Xavier[T]` from `pkg/utils/init.go` for projection init.

### [T-17A03] `pkg/layer/attention/multihead.go` (Backward — ATT-7)

- **Spec:** `l2-attention-impl.md` §4 ATT-7 row + §5.5 softmax-backward helpers
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestAttentionBackwardFD -count=1 ./pkg/layer/attention/` PASS; finite-difference gradient check (max_abs_err < 1e-4 for T=float64) covers `Wq`/`Wk`/`Wv`/`Wo` + biases + input gradient `∂L/∂X`; tested at `Dmodel=8, NumHeads=2, SeqLen=4, Dk=4`
- **Handoff:** Combined with A02 forward, Backward unlocks training-loop integration in A05.
- **Notes:** Four-path decomposition per ATT-7. Softmax-backward Jacobian: `dScores[i, :] = (dA[i, :] - <dA[i, :], A[i, :]>) ⊙ A[i, :]`. Reuse cached `A` (attention weights post-softmax) from forward — do NOT recompute. Scaling factor `1 / sqrt(Dk)` precomputed once in forward, reused in backward.

### [T-17A04] `pkg/layer/attention/masked_layer.go` + cell.go mask helpers

- **Spec:** `l2-attention-impl.md` §5.3 + ATT-5 / ATT-6 / ATT-C5 + l1-attention.md §4.4
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestAttentionMaskCausal -count=1 ./pkg/layer/attention/` PASS — post-softmax weight `A[i, j] = 0` for `j > i` when `Causal: true`; `TestAttentionMaskPadding` PASS — post-softmax weight `A[i, j] = 0` for `m[j] = false` regardless of `i`; both masks compose additively (causal + padding); padding mask reset after Forward consumption (per-call semantics)
- **Handoff:** A05 advertises MaskedLayer interface; pkg/layer/transformer (Phase 18) reuses pattern.
- **Notes:** Causal mask materialized inline per call — O(SeqLen²) cheap write before softmax. Padding mask broadcast over query axis: `scores[i, j] += -Inf if m[j] == false`. Both write `-Inf` (signal to softmax — exp(-Inf) = 0 exactly). `var _ MaskedLayer[float64] = (*MultiHeadAttention[float64])(nil)` separate compile-time assertion.

### [T-17A05] `pkg/layer/attention/multihead.go` JSON + `pkg/nn/options.go` + `compile.go`

- **Spec:** `l2-attention-impl.md` §5.6 + ATT-9 / ATT-10
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestAttentionJSON -count=1 ./pkg/layer/attention/` PASS — round-trip bit-exact for `Wq/Wk/Wv/Wo + biases`; `UnmarshalJSON` validates `len(Wq) == Dmodel*Dmodel` and `Dmodel % NumHeads == 0` per ATT-C3 (returns category error otherwise); `go test -run TestWithAttention -count=1 ./pkg/nn/` PASS — three options compose with Dense/Output; compile() inserts attention prefix after Recurrent layers, before Dense stack
- **Handoff:** B06 follows on same `pkg/nn/options.go` + `compile.go` files; sequence: A05 commits → B06 rebases.
- **Notes:** Padding mask explicitly NOT serialized (runtime-only). Add `AttentionLayers []layer.Layer[T]` field to config; compile() slot mirrors Conv2D prefix pattern (Phase 14 precedent). Three options: `WithAttention(seqLen, dmodel int)` (single-head sugar, `NumHeads=1`), `WithMultiHeadAttention(seqLen, dmodel, numHeads int)`, `WithCausalAttention(seqLen, dmodel, numHeads int)` (sets `Causal: true`). **Sequential dependency: T-17A05 must commit before T-17B06 — both modify `pkg/nn/options.go` + `compile.go`.**

### [T-17B01] `pkg/utils/errors.go` (ErrVocabOutOfRange) + `pkg/layer/core.go` (IDLayer[T])

- **Spec:** `l2-embedding-impl.md` §5.5 + EMB-7 + C32 (error informativeness)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go vet ./...` clean; `errors.Is(err, utils.ErrVocabOutOfRange)` returns true on bounds violation; `errors.Is(utils.ErrVocabOutOfRange, utils.ErrUserConfig)` returns true (sentinel chain per C32); compile-time: every concrete TokenEmbedding/EmbeddingStack satisfies `layer.IDLayer[T]`
- **Handoff:** B03 wraps bounds errors with this sentinel; B05 wires IDLayer assertion.
- **Notes:** Sentinel chain: `var ErrVocabOutOfRange = fmt.Errorf("vocabulary lookup out of range: %w", ErrUserConfig)`. IDLayer extends Layer[T] with `ForwardIDs(ids []int) ([]T, error)`. Both type changes are additive (no breaking change).

### [T-17B02] `pkg/layer/embedding/sparse.go` + `doc.go`

- **Spec:** `l2-embedding-impl.md` §5.4 + EMB-9
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestSparseGrad -count=1 ./pkg/layer/embedding/` PASS — add/iter/reset cycle correctness; bitset population count matches `NumSet`; dense pattern (every row touched) iterates all rows; sparse pattern (5% rows touched) iterates only set rows; coverage line for `sparse.go` ≥90%
- **Handoff:** B03 + B04 use sparseGrad for per-row gradient accumulation.
- **Notes:** Bitset stored as `[]uint64` of length `ceil(Rows / 64)`. `add(row, grad)` sets bit `row` and accumulates into `Buf[row*Dmodel : (row+1)*Dmodel]`. `iter` walks set bits via popcount loop. `reset` clears bitset only (Buf rows overwritten next add).

### [T-17B03] `pkg/layer/embedding/token.go`

- **Spec:** `l2-embedding-impl.md` §5.2 + EMB-1 / EMB-2 / EMB-7 / EMB-8 / EMB-9
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestTokenEmbedding -count=1 ./pkg/layer/embedding/` PASS — lookup correctness for V=10/Dmodel=4 hand-verified fixture; EMB-7 bounds error returns offending position + id + V; sparse backward finite-difference (max_abs_err < 1e-4 for T=float64); JSON round-trip bit-exact; `var _ layer.IDLayer[float64] = (*TokenEmbedding[float64])(nil)` compile-time assertion; coverage line ≥85%
- **Handoff:** B05 (EmbeddingStack) composes with this; B06 wires as input-side layer.
- **Notes:** Lookup is `copy(out[i*Dmodel:(i+1)*Dmodel], W_E[ids[i]*Dmodel:(ids[i]+1)*Dmodel])` per row. No bias term per EMB-2. `Init(rng)` Xavier on dim Dmodel. Forward caches `ids` for backward. Backward returns `dIn = nil` (input is integer, not differentiable).

### [T-17B04] `pkg/layer/embedding/table.go` + `positional.go`

- **Spec:** `l2-embedding-impl.md` §5.3 + EMB-3 / EMB-4 / EMB-5 / EMB-8 + Single-struct-with-Mode pattern
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestSinusoidalFormula -count=1 ./pkg/layer/embedding/` PASS — row-0 is `[0, 1, 0, 1, ...]` exact; reference value match for `(pos=1, Dmodel=4)`: `[sin(1), cos(1), sin(0.01), cos(0.01)]` within 1e-7; `TestLearnablePositional` PASS — Xavier init on dim Dmodel; learnable JSON round-trip bit-exact; `TestSinusoidalNoTableJSON` — sinusoidal envelope contains `{Type, Mode, MaxSeqLen, Dmodel}` but NO `W_P` field; coverage ≥85%
- **Handoff:** B05 composes; B06 may construct from option.
- **Notes:** `buildSinusoidalTable` computes `freq := T(1) / math.Pow(10000, T(2*k)/T(dmodel))`; stores `sin(pos*freq)` at `[pos*Dmodel + 2k]` and `cos(pos*freq)` at `[pos*Dmodel + 2k+1]`. Odd-Dmodel: final dim gets `sin` only. Forward returns `Table[0 : seqLen*Dmodel]` slice — zero copy. Backward no-op for sinusoidal; learnable accumulates dense gradient over `[0, seqLen)` rows via sparseGrad.

### [T-17B05] `pkg/layer/embedding/stack.go`

- **Spec:** `l2-embedding-impl.md` §5.5 + EMB-6
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestEmbeddingStack -count=1 ./pkg/layer/embedding/` PASS — additive composition: `out[i*Dmodel + j] == W_E[ids[i]*Dmodel + j] + Table[i*Dmodel + j]`; backward routes gradient to both children (sparse to token, position-scoped to learnable positional, no-op to sinusoidal); IDLayer compile-time assertion compiles; JSON envelope owns both children
- **Handoff:** B06 attaches as input-side layer to NN; canonical NLP pipeline pattern.
- **Notes:** ForwardIDs calls `tok.ForwardIDs(ids)` then `for i := range out { out[i] += pos.Forward(len(ids))[i] }` — single in-place loop fused with the lookup write. Backward fans out to both children sequentially. JSON envelope: `{Type:"stack", Mode:posMode, Token: {...}, Positional: {...}}`.

### [T-17B06] `pkg/nn/options.go` + `compile.go` + `pkg/optimizer/sparse.go`

- **Spec:** `l2-embedding-impl.md` §5.6 / §5.7 + EMB-9
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestWithEmbedding -count=1 ./pkg/nn/` PASS — three options compose; compile() rejects multiple embedding-layer options via category error; downstream Dmodel match validated; `TestSGDSparseStep` — SGD touches only set rows on synthetic 1000-row table with 5% set; new optimizer sparse path coverage ≥80%
- **Handoff:** Closes Track B; gate T-17Z01 checks compile.
- **Notes:** **Sequential dependency: this task must follow T-17A05 (both modify `pkg/nn/options.go` + `compile.go`).** Add `EmbeddingLayer layer.Layer[T]` field to config; compile() type-asserts input layer to IDLayer[T] — if asserts, the training loop input typing flips from `[][]T` to `[][]int`. New `SparseParamAccessor[T]` extension declared in `pkg/optimizer/sparse.go`. SGD baseline: existing dense path stays default; type-assert layer to SparseParamAccessor[T] inside Step, use sparse iter when present. Adam/RMSProp sparse-moment approximation documented in `sparse_test.go` as known divergence — full sparse-moment support deferred to Phase 18+.

### [T-17T01] Validation Track A

- **Goal:** Verify Attention implementation matches ATT-1..10 invariants against spec.
- **Method:** Run `go test -race -count=1 ./pkg/layer/attention/`. Specific tests: `TestAttentionForward`, `TestAttentionBackwardFD`, `TestAttentionScaleFactor` (ATT-3 entropy bound), `TestAttentionMaskCausal`, `TestAttentionMaskPadding`, `TestAttentionMaskCompose`, `TestAttentionJSON`. Coverage: `go test -cover ./pkg/layer/attention/` ≥85%.
- **Status:** Todo
- **Notes:** ATT-3 scaling-factor entropy test is the highest-leverage correctness check — softmax-output entropy `> 0.5 * ln(SeqLen)` for `Dk = 64` catches the missing-scale-by-sqrt(Dk) bug which is the most common attention divergence cause.

### [T-17T02] Validation Track B

- **Goal:** Verify Embedding implementation matches EMB-1..10 invariants against spec.
- **Method:** Run `go test -race -count=1 ./pkg/layer/embedding/`. Specific tests: `TestTokenEmbedding`, `TestTokenEmbeddingBoundsError`, `TestSinusoidalFormula`, `TestLearnablePositional`, `TestSinusoidalNoTableJSON`, `TestSparseGrad`, `TestEmbeddingStack`. Coverage: ≥85% for `pkg/layer/embedding/`; ≥80% for new `pkg/optimizer/sparse.go`. Verify EMB-7 error message matches expected format: `embedding lookup at position N: id X out of vocabulary [0, V): ...`.
- **Status:** Todo
- **Notes:** EMB-9 sparse gradient is the highest-leverage performance check — verify per-step work is `O(unique IDs × Dmodel)`, not `O(V × Dmodel)`. Add a benchmark `BenchmarkTokenEmbeddingSparseUpdate` comparing batches of 32 unique IDs against the full-V baseline (dense gradient) showing 10× memory reduction on `V=10000`.

### [T-17Z01] Phase 17 release gate

- **Goal:** Confirm Phase 17 v0.15.0 RC ready.
- **Method:** `go build ./...` clean (default tags, no cgo); `go test ./pkg/...` clean (skipping `-race` per existing Windows CGO limitation, documented in Phase 15 STATE.md note); coverage per-package floor verification (`pkg/layer/attention/` ≥85%; `pkg/layer/embedding/` ≥85%; `pkg/optimizer/` ≥80% maintained; `pkg/nn/` ≥75%); CHANGELOG.md v0.15.0 entry written with "NLP Foundation" bullet list; v0.15.0 tag prepared (user runs `git tag -a v0.15.0` per finalization protocol — agent never auto-tags).
- **Status:** Todo
- **Notes:** Phase 18 (Transformer Block implementation) scoping condition: T-17Z01 must close. The frontmatter `provides` field on this file should list `MultiHeadAttention[T]` and `EmbeddingStack[T]` so Phase 18 dependency resolution picks them up. Documented in §6 of l2-transformer-impl.md as a hard prerequisite.
