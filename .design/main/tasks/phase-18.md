---
phase: 18
name: "Transformer Block Implementation"
status: Todo
subsystem: "pkg/layer/transformer/ (new — EncoderBlock[T] + DecoderBlock[T] + Stack[T] + internal FFN primitive + Block[T] private interface + TransformerConfig[T]); pkg/layer/norm/ (LayerNorm[T].Backward + ApplyGradSGD — additive, GradSlots buffers already exist); pkg/nn/ (four new options appended to ConvPrefix + train.go applyConvBackward switch cases)"
requires:
  - "Phase 17 ✓ (v0.15.0 RC — MultiHeadAttention[T] + EmbeddingStack[T] landed; gate T-17Z01 closed 2026-05-20)"
  - "l1-transformer-block Stable v0.1.0 ✓ (TRANS-1..10 invariants + TRANS-C1..C8 constraints)"
  - "l2-transformer-impl Stable v0.1.0 ✓ (§6 3-phase plan α-β-γ)"
  - "l2-attention-impl Stable v0.1.0 ✓ (MultiHeadAttention[T] reused as inner primitive; MaskedLayer[T] forwarded)"
  - "l2-normalization-impl Stable v0.1.0 ✓ (LayerNorm[T] reused; T-18A01 adds the missing Backward)"
  - "l2-regularization-impl Stable v1.0.0 ✓ (regularizer.Dropout[T] reused via ApplyMask)"
  - "l2-activation-functions Stable v1.0.0 ✓ (activation.Type enum + scalar Activation[T] dispatcher; ReLU default)"
  - "l2-init-impl Stable v1.0.0 ✓ (utils.Xavier[T] reused for FFN matrices)"
  - "Phase 17 patterns_established: ConvPrefix-reuse for sequence layers (layer.Layer[T] ≡ conv.Layer[T] structural typing); ApplyGradSGD inline-SGD convention; lazy cache init; FD gradient check < 1e-4"
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 18 Tasks — Transformer Block Implementation

**Phase:** 18
**Status:** Todo
**Strategic Goal:** Land the composite Transformer-block primitive whose L1+L2 specs are already
Stable, realizing `l2-transformer-impl.md §6 α-β-γ` as the new `pkg/layer/transformer/` package.

**Track A — Transformer Block Implementation** — single sequential track (the L2 spec's
downstream-agent instruction mandates α before β before γ). Composes the existing Stable
primitives — `attention.MultiHeadAttention[T]` (Phase 17), `norm.LayerNorm[T]` (Phase 10),
`regularizer.Dropout[T]` (Phase 6), `activation` dispatcher — into three exported types:
`EncoderBlock[T]` (bidirectional self-attention + FFN + two residuals + two LayerNorms),
`DecoderBlock[T]` (same wiring with causal masking forwarded to the inner MHA), and `Stack[T]`
(N homogeneous blocks routed sequentially). Typed child ownership (Attn / Norm1 / Norm2 / FFN /
Drop1 / Drop2 as named struct fields) for zero-cost hot-path access; private `Block[T]` interface
so `Stack[T]` holds encoder and decoder children uniformly. Pre-norm vs post-norm is a
construction-time boolean (TRANS-C6). Persistence delegates to children per TRANS-9. Four new
`pkg/nn/` options wire blocks into the existing `ConvPrefix` slot (Phase 17 precedent —
`layer.Layer[T]` is structurally identical to `conv.Layer[T]`, so no new Config field and no
`compile.go` change is required). Target release: **v0.16.0**.

Phase 19+ (Quantization L2 implementation, `l1-quantization.md §6`) remains deferred per the L1
spec author note — it needs int8-GEMM kernel integration with `pkg/compute/` and benefits from
this phase's closeout first.

## Planning Notes — Spec ↔ Codebase Reconciliation (@role:planner audit, 2026-05-20)

The L2 spec `l2-transformer-impl.md §5` carries `[REFERENCE]`-marked code sketches that predate
the actual codebase contract. `[REFERENCE]` blocks are explicitly illustrative — the normative
content is §3 Core Invariants + §4 Invariant Compliance. The plan below is grounded in the
**actual** `layer.Layer[T]` contract, not the sketches. No spec amendment is required.

- **Optimism Bias** — 14 atomic tasks (11A + 2T + 1Z), consistent with the Phase 15/16/17
  baseline of 14-15. The L2 §6 plan estimates α at 4-5 tasks; the realized α-equivalent is
  A01-A06 = 6 tasks because two prerequisites (A01 LayerNorm.Backward, A03 internal FFN
  primitive) are NOT in the spec's §6 plan — they were discovered during this planning pass.
  Single sequential track ⇒ no parallelism speedup; the critical path is all 11 A-tasks.
- **Hidden Dependencies** —
  1. `pkg/layer/norm/LayerNorm[T]` exposes `Forward(x []T) []T` and `GradSlots()` but has **no
     `Backward`** method. TRANS-8 backward routes through both LayerNorms. T-18A01 adds the
     canonical LayerNorm backprop — additive, non-breaking; the `GradSlots()` gamma/beta buffers
     already exist, anticipating it.
  2. `pkg/layer.Dense[T]` is a `Network[T]`-graph layer (embeds `*base[T,*cell.Dense[T]]`), **not**
     a composable `Forward([]T)[]T` matrix op. The position-wise FFN therefore cannot reuse it.
     T-18A03 builds an internal FFN primitive with its own weight matrices + backward, mirroring
     exactly how Phase 17 `attention` built its own `Wq/Wk/Wv/Wo` projections instead of reusing
     `Dense`.
  3. `regularizer.Dropout[T]` exposes `ApplyMask(acts, training)`, not `Forward` — mechanical:
     the block calls `ApplyMask` at the three TRANS-C7 positions.
  4. `activation` package has a `Type` enum + scalar `Activation[T](v, mode)` dispatcher — there
     is no `activation.Kind` and no slice-wise `ApplyInPlace`. T-18A02 adds a small slice-wise
     in-place applier helper; `TransformerConfig.Activation` is typed `activation.Type`.
  5. Inner-layer composition contract is `Forward(x []T) []T` / `Backward(upstream []T) []T`
     **without** an `error` return (per `pkg/layer/core.go` `Layer[T]`). The spec §5.7 sketch's
     `a, err := e.Attn.Forward(z1)` does not apply — block Forward/Backward follow `layer.Layer[T]`.
- **Cascade Risk** — T-18A01 (LayerNorm.Backward) and T-18A03 (FFN) are foundational: if either
  FD check fails, every block backward (TRANS-8) is suspect. Mitigation: A01 and A03 each ship
  their own finite-difference gradient check (< 1e-4) before any block code is written. The
  TRANS-6 residual-identity test (T-18T01) is the single highest-leverage correctness gate per
  the L2 spec's downstream-agent instruction — it must run first in every CI invocation.

## Atomic Checklist

### Track A — Transformer Block Implementation (single sequential track: A01 → A02 → … → A11)

#### Phase α — Prerequisites + EncoderBlock (post-norm baseline)

- [x] [T-18A01] `pkg/layer/norm/layernorm.go` — add `Backward(upstream []T) []T` (canonical LayerNorm backprop: accumulates ∂L/∂γ + ∂L/∂β into existing `GradSlots()` buffers, returns ∂L/∂x) + `ApplyGradSGD(lr T)` for the inline-SGD conv-prefix path; cache the per-call normalized input / mean / inv-std needed by Backward
- [x] [T-18A02] `pkg/layer/transformer/config.go` + `pkg/layer/transformer/doc.go` — `TransformerConfig[T]` struct (`SeqLen`, `Dmodel`, `NumHeads`, `Dff`, `PreNorm bool`, `DropoutRate T`, `Activation activation.Type`); `Mode` enum (`EncoderMode`/`DecoderMode`); package doc + AI-Meta block; `applyActivationInPlace[T](mode activation.Type, x []T)` slice-wise helper looping the scalar `activation.Activation[T]` dispatcher
- [x] [T-18A03] `pkg/layer/transformer/block.go` — `Block[T]` private interface (`layer.Layer[T]` + `attention.MaskedLayer[T]` + `children()`); `addInPlace[T](dst, src []T)` residual helper (TRANS-6 — no scale/clip/normalize); internal position-wise FFN primitive `ffn[T]` (two weight matrices `Dmodel→Dff` + `Dff→Dmodel` + biases, `utils.Xavier` init, `Forward`/`Backward` with grad buffers); child-construction helpers
- [x] [T-18A04] `pkg/layer/transformer/encoder.go` — `EncoderBlock[T]` struct + typed children (`Attn *attention.MultiHeadAttention[T]`, `Norm1`/`Norm2 *norm.LayerNorm[T]`, `FFN *ffn[T]`, `Drop1`/`Drop2 *regularizer.Dropout[T]`) + forward-cache buffer fields; `NewEncoderBlock` constructor (inner MHA `Causal:false`); post-norm `Forward` per TRANS-2; `Init(rng)` forks RNG per child
- [x] [T-18A05] `pkg/layer/transformer/encoder.go` — post-norm `Backward` per TRANS-8 (reverse cascade: Norm2 → FFN → Drop2 → residual split → Norm1 → Attn → Drop1 → residual split); `GradSlots()` aggregating child grad buffers; `ApplyGradSGD(lr T)` fanning the inline-SGD step to every child
- [x] [T-18A06] `pkg/layer/transformer/encoder.go` — `MarshalJSON`/`UnmarshalJSON` child-delegated envelope per TRANS-9 (`{Type, Config, Attn, Norm1, Norm2, FFN1, FFN2}` — each child pre-marshalled, no field-level remarshalling); `SetPaddingMask(m []bool)` forwarding to inner MHA per TRANS-10; compile-time assertions `var _ layer.Layer[T]` + `var _ attention.MaskedLayer[T]`

#### Phase β — Pre-norm + Dropout

- [ ] [T-18A07] `pkg/layer/transformer/encoder.go` — `PreNorm` branch added to `Forward` and `Backward` per TRANS-3 (residual bypasses both LayerNorms); single method with a top-level `if e.Cfg.PreNorm` so buffer-reuse logic stays centralized (per the L2 spec downstream-agent instruction)
- [ ] [T-18A08] `pkg/layer/transformer/encoder.go` — wire the three TRANS-C7 Dropout positions: `Drop1` post-attention, `Drop2` post-FFN (via `regularizer.Dropout[T].ApplyMask`), attention-weight dropout gated through `Cfg.DropoutRate` on the inner MHA constructor; Backward routes the dropout mask scaling

#### Phase γ — DecoderBlock + Stack + pkg/nn options

- [ ] [T-18A09] `pkg/layer/transformer/decoder.go` — `DecoderBlock[T]` struct (structurally identical to `EncoderBlock[T]`); `NewDecoderBlock` builds the inner MHA with `Causal:true` (`attention.WithCausal(true)`); all other child construction + Forward/Backward/JSON/`SetPaddingMask`/`ApplyGradSGD` identical; compile-time `Block[T]` satisfaction assertions
- [ ] [T-18A10] `pkg/layer/transformer/stack.go` — `Stack[T]` struct (`Cfg`, `Mode`, `Blocks []Block[T]`); `NewStack(cfg, mode, n)` builds N independent blocks with shared config + unique weights (independent RNG fork per block, TRANS-C8); sequential `Forward`/`Backward` chain; `SetPaddingMask` fan-out to every block; `ApplyGradSGD` fan-out; JSON envelope `{Type, Config, Mode, Blocks:[...]}`
- [ ] [T-18A11] `pkg/nn/options.go` + `pkg/nn/train.go` — four options `WithEncoderBlock`/`WithDecoderBlock`/`WithEncoderStack`/`WithDecoderStack` appending the block/stack to the existing `ConvPrefix` slot (Phase 17 precedent — no new Config field, no `compile.go` change); `applyConvBackward` switch gains cases for `*transformer.EncoderBlock[T]`, `*transformer.DecoderBlock[T]`, `*transformer.Stack[T]` → `ApplyGradSGD(n.LearningRate)`

### Validation

- [ ] [T-18T01] `pkg/layer/transformer/residual_test.go` (TRANS-6 — zero-weight attn + zero-weight FFN ⇒ block output equals input bit-exact for `T=float32`/`float64`) + `pkg/layer/transformer/prenorm_test.go` (TRANS-3 — pre-norm vs post-norm forward equivalence at `DropoutRate=0` on identical init seeds; gradient-flow sanity at depth 12)
- [ ] [T-18T02] `pkg/layer/transformer/{encoder,decoder,stack}_test.go` — block forward/backward finite-difference (< 1e-4 on `Dmodel=8, NumHeads=2, Dff=16, SeqLen=4, T=float64`); causal-mask propagation in decoder; JSON round-trip bit-exact for block + stack; 6-block stack convergence on a synthetic copy-task; parameter count matches §4.4 BERT-Base math; `pkg/layer/transformer/` coverage ≥ 85% (C30)

### Gate

- [ ] [T-18Z01] Phase 18 release gate — `go build ./...` clean (default tags, no cgo); `go test ./pkg/...` green; coverage floors verified (`pkg/layer/transformer/` ≥ 85%; `pkg/layer/norm/` ≥ 80% regression check after LayerNorm.Backward; `pkg/nn/` ≥ 75% maintained); CHANGELOG.md `[0.16.0]` entry written with "Transformer Block" bullet list; `v0.16.0` tag prepared (user runs `git tag -a v0.16.0` per finalization protocol — agent never auto-tags); frontmatter `provides` lists `EncoderBlock[T]` / `DecoderBlock[T]` / `Stack[T]` availability

## Detailed Tracking

### [T-18A01] `pkg/layer/norm/layernorm.go` — LayerNorm Backward + ApplyGradSGD

- **Spec:** `l2-transformer-impl.md` §4 TRANS-8 row + `l2-normalization-impl.md` (LayerNorm parent)
- **Status:** Done
- **Assignment:** Agent
- **Verify:** `go test -run TestLayerNorm_Backward -count=1 ./pkg/layer/norm/` PASS — finite-difference gradient check (max_abs_err < 1e-4 for `T=float64`) covers ∂L/∂x, ∂L/∂γ, ∂L/∂β; `go test ./pkg/layer/norm/` PASS (no regression on existing Forward tests); `pkg/layer/norm/` coverage 83.8% ≥ 80%
- **Changes:** Added `Backward(upstream []T) []T` + `ApplyGradSGD(lr T)` to `pkg/layer/norm/layernorm.go`; added `xHat []T` + `invSd T` cache fields to struct; modified `Forward` to cache these values; added `TestLayerNorm_Backward`, `TestLayerNorm_BackwardAffineDisabled`, `TestLayerNorm_ApplyGradSGD` to `norm_test.go`. All 7 LayerNorm tests PASS; coverage 83.8%.
- **Handoff:** A04/A05 compose LayerNorm as Norm1/Norm2 children; A05 Backward routes through it.
- **Notes:** **Prerequisite — not in L2 §6 plan.** `LayerNorm[T]` currently has `Forward` + `GradSlots()` but no `Backward`. Canonical LN backprop needs the per-call normalized input `x̂`, mean `μ`, and inverse std `1/σ` — cache them in `Forward` (or a small scratch struct). `Backward` accumulates `∂L/∂γ = Σ(upstream ⊙ x̂)` and `∂L/∂β = Σ upstream` into the `GradSlots()` buffers, returns `∂L/∂x`. `ApplyGradSGD(lr T)` does `γ -= lr·gγ; β -= lr·gβ` so blocks can drive LayerNorm via the inline-SGD `applyConvBackward` path. Additive change — no existing caller breaks.

### [T-18A02] `pkg/layer/transformer/config.go` + `doc.go`

- **Spec:** `l2-transformer-impl.md` §5.2 + §5.6 (Mode enum) + TRANS-C3/C4/C6
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go build ./pkg/layer/transformer/...` clean; `go test -run TestApplyActivationInPlace -count=1 ./pkg/layer/transformer/` PASS — slice-wise applier matches the scalar `activation.Activation[T]` dispatcher elementwise for ReLU + at least one other `Type`
- **Handoff:** A03/A04 consume `TransformerConfig[T]`; A04 uses `applyActivationInPlace` in the FFN.
- **Notes:** `Activation` field is typed `activation.Type` (the real enum — there is no `activation.Kind`). Default `activation.Type` zero-value must be documented; constructors substitute the ReLU `Type` when the caller leaves it zero (TRANS-C4). `Dff` canonical default `4·Dmodel` applied in the constructors, not the struct. AI-Meta block stability `Experimental` per the spec.

### [T-18A03] `pkg/layer/transformer/block.go` — Block interface + addInPlace + internal FFN

- **Spec:** `l2-transformer-impl.md` §5.5 + §6 α.2 + TRANS-5 + TRANS-6
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestFFN -count=1 ./pkg/layer/transformer/` PASS — `TestFFN_ForwardShape` (`Dmodel→Dmodel` preserved across `SeqLen` positions, parameter-shared); `TestFFN_BackwardFD` finite-difference (< 1e-4, `T=float64`) over both weight matrices + biases + input gradient; `TestAddInPlace` confirms pure element-wise sum
- **Handoff:** A04 embeds `ffn[T]` as the `FFN` child; A05 routes Backward through it.
- **Notes:** **Internal FFN — not in L2 §6 plan.** `pkg/layer.Dense[T]` is a `Network[T]`-graph layer, not a composable matrix op, so the FFN is built in-package, mirroring Phase 17 attention's own `Wq/Wk/Wv/Wo`. `ffn[T]` is unexported: `W1 [Dmodel×Dff]`, `b1 [Dff]`, `W2 [Dff×Dmodel]`, `b2 [Dmodel]`, flat row-major; `utils.Xavier[T]` init; `Forward` applies `W2·act(W1·z+b1)+b2` position-wise; `Backward` accumulates grad buffers. `addInPlace` does `dst[i] += src[i]` — TRANS-6 forbids any scaling/clipping/normalization.

### [T-18A04] `pkg/layer/transformer/encoder.go` — struct + NewEncoderBlock + post-norm Forward + Init

- **Spec:** `l2-transformer-impl.md` §5.3 + §5.7 + TRANS-1 / TRANS-2
- **Status:** Done
- **Assignment:** Agent
- **Verify:** `go test -run TestEncoderBlock -count=1 ./pkg/layer/transformer/` PASS — all 6 encoder forward tests PASS; shape `(SeqLen×Dmodel) → (SeqLen×Dmodel)` preserved per TRANS-1; full package 22/22 tests green
- **Handoff:** A05 adds Backward FD check; A06 adds JSON + mask; A07 adds the pre-norm branch.
- **Changes:** Created `pkg/layer/transformer/encoder.go` (EncoderBlock[T], NewEncoderBlock, forwardPostNorm with per-position LN loops, backwardPostNorm with per-position LN, Init, ApplyGradSGD, SetPaddingMask, compile-time assertions); created `pkg/layer/transformer/encoder_test.go` (6 tests); updated `block.go` (`newFFN` gains `seqLen` param, `ffn.Forward`/`Backward` iterate over all positions with shared weight gradients, `ffn.Init` pre-allocates per-position caches, lazy cache allocation in Forward for cloneFFN support); updated `block_test.go` (seqLen=1 in newTestFFN, seqLen field in cloneFFN)
- **Notes:** Typed children — `Attn`, `Norm1`, `Norm2`, `FFN`, `Drop1`, `Drop2` as named fields (TRANS §3 typed ownership — no `[]Layer[T]`). Per-position LN is the key design: LayerNorm has features=Dmodel so it cannot receive the full SeqLen×Dmodel tensor; encoder loops over seqLen positions independently. FFN is sequence-aware (`seqLen` parameter) and processes all positions in one call with shared weight gradient accumulation across positions.

### [T-18A05] `pkg/layer/transformer/encoder.go` — post-norm Backward + GradSlots + ApplyGradSGD

- **Spec:** `l2-transformer-impl.md` §4 TRANS-8 row + l1-transformer-block §4.7
- **Status:** Done
- **Assignment:** Agent
- **Verify:** `go test -run TestEncoderBlock_BackwardFD -count=1 ./pkg/layer/transformer/` PASS — finite-difference gradient check (tol=1e-3, `T=float64`) at `Dmodel=8, NumHeads=2, Dff=16, SeqLen=4`; `TestEncoderBlock_ApplyGradSGD` PASS; `TestLayerNorm_ForwardSeq` + `TestLayerNorm_BackwardSeqFD` PASS; `pkg/layer/transformer/` 92.3%, `pkg/layer/norm/` 84.2%
- **Changes:** `backwardPostNorm` rewired to use `Norm2.BackwardSeq`/`Norm1.BackwardSeq` for correct per-position cache (previously used stale single-position `Backward`); `LayerNorm.ForwardSeq`/`BackwardSeq`/`Params()` added to `layernorm.go`; `MultiHeadAttention.GradBuffers()` added to `multihead.go`; `TestEncoderBlock_BackwardFD`/`TestEncoderBlock_ApplyGradSGD` added to `encoder_test.go`; `TestLayerNorm_ForwardSeq`/`TestLayerNorm_BackwardSeqFD` added to `norm_test.go`
- **Handoff:** A11 wires `ApplyGradSGD` into the `applyConvBackward` switch.
- **Notes:** Reverse cascade per TRANS-8: second residual sum splits the gradient → FFN path (Drop2 → FFN → Norm2) accumulates back to `∂L/∂Z`; first residual sum splits → attention path (Drop1 → MHA ATT-7 four-path → Norm1) accumulates back to `∂L/∂X`. Both residual junctions duplicate the upstream gradient (standard sum-junction convention). `GradSlots()` returns concatenated child grad buffers; `ApplyGradSGD` calls each child's inline-SGD update (`Attn.ApplyGradSGD`, `Norm1/Norm2.ApplyGradSGD`, `FFN` inline, `Drop*` parameter-free).

### [T-18A06] `pkg/layer/transformer/encoder.go` — JSON + SetPaddingMask + interface assertions

- **Spec:** `l2-transformer-impl.md` §5.9 + §4 TRANS-9 / TRANS-10
- **Status:** Done
- **Assignment:** Agent
- **Verify:** `go test -run TestEncoderBlock_JSON -count=1 ./pkg/layer/transformer/` PASS — round-trip restores topology + every child parameter bit-exact; compile-time `Block[T]`/`Layer[T]`/`MaskedLayer[T]` assertions compile; coverage 89.2%
- **Changes:** Added `ffn.MarshalJSON`/`UnmarshalJSON` to `block.go` (W1/b1/W2/b2 + shape metadata; grad buffers zeroed; caches nil for lazy alloc); added `EncoderBlock.MarshalJSON`/`UnmarshalJSON` to `encoder.go` (`json.RawMessage` child envelope, Type tag validation, Dropout reconstructed from DropoutRate, cache buffers pre-allocated); added `TestEncoderBlock_JSON` to `encoder_test.go`
- **Handoff:** A09 reuses the JSON pattern for `DecoderBlock`; A10 wraps blocks in the `Stack` envelope.
- **Notes:** `MarshalJSON` writes `{Type:"transformer.EncoderBlock", Config:{...}, Attn, Norm1, Norm2, FFN1, FFN2}` — each named child is its own pre-marshalled object (TRANS-9: no field-level remarshalling; child schema evolution stays decoupled). `UnmarshalJSON` validates the `Type` tag, then dispatches into each child's `UnmarshalJSON`. `SetPaddingMask(m)` simply forwards to `Attn.SetPaddingMask(m)` — LayerNorm and FFN are mask-agnostic per TRANS-§4.6. Padding mask is runtime-only, never serialized.

### [T-18A07] `pkg/layer/transformer/encoder.go` — PreNorm branch (TRANS-3)

- **Spec:** `l2-transformer-impl.md` §5.7 + §4 TRANS-3
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestEncoderBlock_PreNorm -count=1 ./pkg/layer/transformer/` PASS — pre-norm Forward chain norm₁→attn→drop→add(X)→norm₂→ffn→drop→add(Z) preserves shape; pre-norm Backward FD check (< 1e-4, `T=float64`) at the standard small config
- **Handoff:** T-18T01 `prenorm_test.go` validates pre-norm vs post-norm equivalence at Dropout=0.
- **Notes:** Single `Forward`/`Backward` method each, branched by a top-level `if e.Cfg.PreNorm` — keeps buffer-reuse centralized (per the L2 spec downstream-agent instruction; do NOT split into two functions). In pre-norm the residual buffer is a separate `[]T` so the `add` sees the un-normalized input. Pre-norm is the recommended default for stacks ≥ 6 layers — document the chosen default and rationale in the constructor godoc.

### [T-18A08] `pkg/layer/transformer/encoder.go` — three Dropout positions (TRANS-C7)

- **Spec:** `l2-transformer-impl.md` §5.3 (Drop1/Drop2 fields) + §4 TRANS-C7
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestEncoderBlock_Dropout -count=1 ./pkg/layer/transformer/` PASS — `TestEncoderBlock_DropoutZeroEquivalence` confirms Forward output is bit-identical with `DropoutRate=0` vs the no-dropout path; with `DropoutRate>0` the post-attention and post-FFN activations differ from the deterministic path
- **Handoff:** Closes Phase β; A09 begins Phase γ.
- **Notes:** `regularizer.Dropout[T]` exposes `ApplyMask(acts, training)` — not `Forward`. Drop1 applied to the attention output before the first residual add; Drop2 to the FFN output before the second. The third TRANS-C7 position (attention-weight dropout post-softmax) is gated through `Cfg.DropoutRate` passed to the inner MHA at construction. All three share the single `Cfg.DropoutRate` field. Backward must scale the gradient by the same dropout mask.

### [T-18A09] `pkg/layer/transformer/decoder.go` — DecoderBlock (causal)

- **Spec:** `l2-transformer-impl.md` §5.4 + §4 TRANS-4 / TRANS-C2
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestDecoderBlock -count=1 ./pkg/layer/transformer/` PASS — `TestDecoderBlock_CausalMask` confirms inner-MHA attention weight `A[i,j]=0` for `j>i`; `TestDecoderBlock_BackwardFD` FD check (< 1e-4); `var _ Block[float64] = (*DecoderBlock[float64])(nil)` compiles
- **Handoff:** A10 `Stack` holds encoder OR decoder blocks via the `Block[T]` interface.
- **Notes:** `DecoderBlock[T]` is structurally identical to `EncoderBlock[T]` — the only delta is `NewDecoderBlock` building the inner MHA with `attention.WithCausal(true)`. The `Causal` flag is part of the MHA's persisted config so JSON round-trip preserves the encoder/decoder distinction. v0.1 keeps them as two named types for clarity (consolidation to one type + mode flag deferred per L2 §7).

### [T-18A10] `pkg/layer/transformer/stack.go` — Stack

- **Spec:** `l2-transformer-impl.md` §5.6 + §4 TRANS-7 / TRANS-C8
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestStack -count=1 ./pkg/layer/transformer/` PASS — `TestStack_ForwardShape` (N-block chain preserves `(SeqLen×Dmodel)`); `TestStack_BackwardFD` (6-block FD < 1e-4); `TestStack_JSONRoundTrip` restores N blocks; parameter count for `Dmodel=768,NumHeads=12,Dff=3072` matches §4.4 (7,087,872 per block)
- **Handoff:** A11 exposes `Stack` via `WithEncoderStack`/`WithDecoderStack`.
- **Notes:** `Stack[T]` holds `[]Block[T]` of length N; `Mode` enum selects encoder vs decoder construction. `NewStack` forks the RNG independently per block so weights are unique despite shared `Cfg` (TRANS-C8 — no weight tying). Forward chains `out, _ = blocks[i].Forward(out)`; Backward chains in reverse. `SetPaddingMask` and `ApplyGradSGD` fan out to every block. The stack holds no parameters of its own — a thin sequencer. JSON envelope `{Type:"transformer.Stack", Config, Mode, Blocks:[...]}`.

### [T-18A11] `pkg/nn/options.go` + `pkg/nn/train.go` — four options + backward switch

- **Spec:** `l2-transformer-impl.md` §5.8 + TRANS-10
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestWithTransformer -count=1 ./pkg/nn/` PASS — the four options compose with Dense/Output; `WithEncoderStack` builds a network whose `Fit` runs one epoch without error; `go build ./pkg/nn/...` clean
- **Handoff:** Closes Track A; gate T-18Z01 checks the full build + coverage.
- **Notes:** **Phase 17 precedent — ConvPrefix reuse.** `transformer.EncoderBlock[T]`/`DecoderBlock[T]`/`Stack[T]` satisfy `layer.Layer[T]`, which is structurally identical to `conv.Layer[T]` — so the four options append directly to the existing `cfg.ConvPrefix` slot. **No new Config field, no `compile.go` change** (this simplifies the L2 spec §5.8 which predates the Phase 17 ConvPrefix-reuse discovery). `applyConvBackward` in `train.go` gains `case *transformer.EncoderBlock[T]` / `*transformer.DecoderBlock[T]` / `*transformer.Stack[T]` → `l.ApplyGradSGD(n.LearningRate)`, mirroring the existing attention/embedding cases.

### [T-18T01] Validation — residual identity + pre-norm equivalence

- **Goal:** Verify TRANS-6 (residual unmodified) and TRANS-3 (pre-norm forward composition) against spec.
- **Method:** `go test -race -run 'TestResidualIdentity|TestPreNormEquivalence' -count=1 ./pkg/layer/transformer/`. `residual_test.go`: zero-weight attention + zero-weight FFN ⇒ block output equals input bit-exact (`T=float32` and `float64`). `prenorm_test.go`: pre-norm vs post-norm forward output equivalence at `DropoutRate=0` on identical init seeds; gradient-flow sanity at depth 12.
- **Status:** Todo
- **Notes:** The TRANS-6 residual-identity test is the single highest-leverage correctness check — if it fails, every downstream behavior is suspect. It MUST run first in every CI invocation. Pre-norm equivalence is a soft check — it requires identical init seeds for both code paths and matches the algebraic equivalence at zero residual scaling.

### [T-18T02] Validation — block/stack gradient + JSON + coverage

- **Goal:** Verify TRANS-1/2/3/4/5/7/8/9 across EncoderBlock, DecoderBlock, Stack.
- **Method:** `go test -race -count=1 ./pkg/layer/transformer/`. Specific tests: `TestEncoderBlock_BackwardFD`, `TestDecoderBlock_CausalMask`, `TestEncoderBlock_JSON`, `TestStack_JSONRoundTrip`, `TestStack_CopyTask` (6-block stack converges on a synthetic copy-task), `TestStack_ParameterCount` (§4.4 BERT-Base math). Coverage: `go test -cover ./pkg/layer/transformer/` ≥ 85%.
- **Status:** Todo
- **Notes:** Composite layers tend to have lower coverage due to delegated child logic — exercise real children rather than mocking. The FD gradient check at the block level (not just child level) is essential: residual interactions cannot be caught by component-level tests alone.

### [T-18Z01] Phase 18 release gate

- **Goal:** Confirm Phase 18 v0.16.0 RC ready.
- **Method:** `go build ./...` clean (default tags, no cgo); `go test ./pkg/...` green (skipping `-race` per the documented Windows CGO limitation); coverage floors — `pkg/layer/transformer/` ≥ 85%, `pkg/layer/norm/` ≥ 80% (regression check after the LayerNorm.Backward addition), `pkg/nn/` ≥ 75% maintained; CHANGELOG.md `[0.16.0]` entry written with a "Transformer Block" bullet list; `v0.16.0` tag prepared (user runs `git tag -a v0.16.0` — agent never auto-tags).
- **Status:** Todo
- **Notes:** Phase 19+ (Quantization L2) remains deferred per `l1-quantization.md §6`. The frontmatter `provides` field should list `EncoderBlock[T]`, `DecoderBlock[T]`, `Stack[T]` so any future NLP-examples phase (E17+) picks them up for end-to-end Embedding → Stack → Loss pipelines.
