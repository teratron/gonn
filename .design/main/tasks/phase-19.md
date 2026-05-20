---
phase: 19
name: "Quantization L2 Implementation"
status: Todo
subsystem: "pkg/quantization/ (new — config.go, params.go, quantizer.go, dense.go, conv.go, attn.go, calibration.go, persistence.go, evaluator.go, doc.go + *_test.go); no changes to pkg/nn/, pkg/layer/, or pkg/compute/ in v0.1 — quantization is a peer transformation layer"
requires:
  - "Phase 18 ✓ (v0.16.0 RC — EncoderBlock[T] + DecoderBlock[T] + Stack[T] landed; gate T-18Z01 closed)"
  - "l1-quantization.md Stable v0.1.0 ✓ (QUANT-1..10 invariants + QUANT-C1..C8 constraints; affine scheme, per-channel granularity, calibration protocol, .qnn.json schema)"
  - "l2-quantization-impl.md Stable v0.1.0 ✓ (§5 three-phase plan α-β-γ; GC-1..GC-5 Go-specific constraints; 10-file package layout)"
  - "l2-layer-types.md Stable v0.1.0 ✓ (Dense/Conv1D/Conv2D weight layouts — quantization targets Phase α)"
  - "l2-attention-impl.md Stable v0.1.0 ✓ (MultiHeadAttention[T] Wq/Wk/Wv/Wo projections — quantization target Phase γ)"
  - "l2-persistence-impl.md Stable v1.0.0 ✓ (.nn.json type-discriminator pattern reused for .qnn.json)"
  - "l2-transformer-impl.md Stable v0.1.0 ✓ (Phase 18 must close before Phase γ attention-projection quantization)"
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 19 Tasks — Quantization L2 Implementation

**Phase:** 19
**Status:** Todo
**Strategic Goal:** Deliver `pkg/quantization/` — a peer transformation layer that consumes a
trained `*nn.Network[T]` and produces a deployable `QuantizedNetwork[T]` int8 artifact persisted
as `.qnn.json`. Implements all 10 L1 invariants (QUANT-1..10) from `l1-quantization.md` across
three deliverable phases: α (weight-only Dense/Conv), β (activation calibration + full-int8 path),
γ (attention projection matrices). Target release: **v0.17.0**.

**Track A — Quantization L2 Implementation** — single sequential track following the L2 §5
three-phase plan. Phase α delivers a minimal deployable subset (weight-only, no calibration
needed). Phase β adds `CalibrationRunner[T]` + full-int8 GEMM path. Phase γ adds attention
projection quantization. No changes to `pkg/nn/`, `pkg/layer/`, or `pkg/compute/` — quantization
is a peer package in the same module (GC-3 read-only network input; GC-5 no `layer.Layer[T]`
for quantized types). Three exported top-level functions: `Quantize[T]`, `Load[T]`, `Evaluate[T]`.

## Planning Notes — Spec ↔ Codebase Reconciliation (@role:planner audit, 2026-05-20)

- **Optimism Bias** — 14 atomic tasks (11A + 2T + 1Z), consistent with the Phase 15/16/17/18
  baseline of 14-15. Phase α (A01-A06) is the deployable gate; β (A07-A09) and γ (A10-A11) are
  additive. No cross-phase compile dependencies within Phase 19 — each α task ships independently.
- **Hidden Dependencies** —
  1. `Quantize[T]` accesses layer weights via type-switch on `net.Config().ConvPrefix` (same pattern
     as `applyConvBackward` in `pkg/nn/train.go`). The public `Network[T]` API does not expose
     raw weight slices — direct struct access per layer type is required. `[NN-CONFIG]` canonical
     reference documents this coupling as acceptable (peer package, same module, GC-3).
  2. `roundHalfEven` is NOT in the Go standard library — T-19A02 implements it in `params.go`
     per L2 §6.1: `math.Floor(x + 0.5)` with special case for `x + 0.5` being an integer.
     The golden-value test in T-19T01 uses the L1 §4.2 example (scale≈0.0236, q(1.0)=42)
     as the normative verification of this implementation.
  3. `CalibrationRunner[T]` (T-19A07) needs to drive the source network's Forward pass to
     collect per-layer activation statistics. In WeightOnly mode (Phase α) calibration samples
     are `nil` — this must be explicitly guarded in `Quantize[T]` to avoid nil-slice panic.
  4. Full-int8 GEMM (T-19A08) accumulates into `int32` — Go has no SIMD intrinsics; the reference
     implementation uses a plain inner loop with `int32` accumulation. Backend kernel extension
     point is documented in AI-Meta but NOT wired (GC-1 no external deps; QUANT-9 pure Go v0.1).
  5. `QuantizedAttentionProjections[T]` (T-19A10) quantizes only Wq/Wk/Wv/Wo projection weights
     accessed via `MultiHeadAttention[T]`'s public fields; scoring + softmax remain float
     (QUANT-10). Phase γ gated on Phase 18 closeout (T-18Z01).
- **Cascade Risk** — T-19A02 (`roundHalfEven` + affine math) is the most critical primitive:
  if the golden-value test fails, every quantized weight is wrong. A01 (config enums) + A02
  (affine math) MUST be verified before writing A03-A06. The L1 §4.2 example value
  `q(1.0) = round((1.0/0.0236) + 0) = round(42.37) = 42` (roundHalfEven) is the single
  normative anchor. Persistence round-trip (T-19A06 + T-19T01) is the second highest-leverage
  gate — int8 bytes must survive base64 encode/decode bit-exact.

## Atomic Checklist

### Track A — Phase α: Weight-Only Int8 (Dense, Conv1D, Conv2D) (A01 → A02 → A03 → (A04 ∥ A05) → A06)

- [ ] [T-19A01] `pkg/quantization/config.go` + `pkg/quantization/doc.go` — `QuantMode` enum (`WeightOnly=0`, `FullInt8=1`); `QuantizationConfig[T utils.Float]` struct (`Mode`, `WeightGranularity`, `ActGranularity`, `Strategy`, `PercentileThreshold`, `MinCalibSamples`, `Seed`); `CalibrationProvenance` struct; default-filling constructor `DefaultQuantizationConfig[T]()`; package doc + AI-Meta block (`Stability: Experimental`)
- [ ] [T-19A02] `pkg/quantization/params.go` — `Granularity` enum (`PerTensor=0`, `PerChannel=1`); `CalibStrategy` enum (`MinMax=0`, `Percentile99p9=1`, `Entropy=2`); `QuantizationParams` struct (`Scale []float64`, `ZeroPoint []int32`, `Granularity`, `Strategy`); pure functions: `dequantize(q int8, p QuantizationParams, ch int) float64`; `quantize(r float64, p QuantizationParams, ch int) int8` using `roundHalfEven`; `roundHalfEven(x float64) int64`; `computeSymmetric(rMin, rMax float64) QuantizationParams`; `computeAsymmetric(rMin, rMax float64) QuantizationParams`; degenerate guard `rMax−rMin < 1e-6` → `scale=1.0, zp=0`
- [ ] [T-19A03] `pkg/quantization/quantizer.go` — `quantizedLayer[T]` private interface (`Forward([]T) []T`); `QuantizedNetwork[T]` struct (`Config QuantizationConfig[T]`, `Layers []quantizedLayer[T]`, `BaselineHash string`, `CalibProvenance CalibrationProvenance`); `Quantize[T](net *nn.Network[T], calibSamples [][]T, cfg QuantizationConfig[T]) (*QuantizedNetwork[T], error)` — reads `net.Config().ConvPrefix` in order via type-switch (`*dense.Dense[T]`, `*conv.Conv1D[T]`, `*conv.Conv2D[T]` cases); `quantizeWeightsTensor(w []float64, cout, cin int, gran Granularity) ([]int8, QuantizationParams)` per QUANT-3; `QuantizedNetwork[T].Forward(x []T) []T` chains through `Layers`; `Load[T](path string) (*QuantizedNetwork[T], error)` wraps `UnmarshalQNN`; SHA-256 `BaselineHash` via `crypto/sha256` over canonical JSON bytes of source network
- [ ] [T-19A04] `pkg/quantization/dense.go` — `QuantizedDense[T]` struct (`WeightParams QuantizationParams`, `Weights []int8`, `Bias []float64`, `Cout, Cin int`); weight-only `Forward(x []T) []T`: for each output channel `c`, reconstruct `w_c[k] = scale[c]·(W_int8[c·Cin+k] − zp[c])` as `float64`, accumulate dot-product with `x`, add `Bias[c]`, cast to `T`; `newQuantizedDense[T](weights []float64, bias []float64, cout, cin int, cfg QuantizationConfig[T]) *QuantizedDense[T]` constructor
- [ ] [T-19A05] `pkg/quantization/conv.go` — `QuantizedConv1D[T]` and `QuantizedConv2D[T]` structs (weight-only, analogous to `QuantizedDense[T]` but with spatial loop over kernel positions); `Forward` dequantizes filter weights on the fly; constructors `newQuantizedConv1D` / `newQuantizedConv2D` reading filter shape from Conv1D/Conv2D structs via direct field access
- [ ] [T-19A06] `pkg/quantization/persistence.go` — `QuantizedNetworkJSON` wire format (all fields per l2-quantization-impl §4.6); `QuantizedLayerJSON` per-layer record; `MarshalQNN[T](qnet *QuantizedNetwork[T]) ([]byte, error)` — encodes int8 weights as base64(`encoding/base64.StdEncoding`), bias as base64 int32 bytes (little-endian); `UnmarshalQNN[T](data []byte) (*QuantizedNetwork[T], error)` — validates `"Type":"quantized"` discriminator, decodes int8+int32 base64 bit-exact; round-trip invariant: integer bytes survive encode/decode with zero float intermediates; `(*QuantizedNetwork[T]).Save(path string) error` writes JSON to disk

### Track A — Phase β: Activation Calibration + Full-Int8 Path (A07 → A08 → A09)

- [ ] [T-19A07] `pkg/quantization/calibration.go` — `layerStats` struct (`min, max float64`, `histogram []int64`, `count int`); `CalibrationRunner[T]` struct (`net *nn.Network[T]`, `cfg QuantizationConfig[T]`, `stats map[int]layerStats`); `Run(samples [][]T) error` — passes all samples through `net.Forward`, records per-layer activation min/max + histogram; hard floor 32 samples returns `ErrCalibTooFewSamples` (pkg/utils C32 error sentinel); soft floor 100 samples logs warning; `Params() (map[int]QuantizationParams, error)` — dispatches `MinMax` (→ `computeSymmetric/Asymmetric`) or `Percentile99p9` (99.9th percentile of |A| from histogram → clipped range → same compute functions); degenerate range guard → `scale=1.0, zp=0` + wrap `ErrDegenerateRange`; new error sentinels `ErrCalibTooFewSamples`, `ErrDegenerateRange` in `pkg/utils/errors.go`
- [ ] [T-19A08] `pkg/quantization/dense.go` (amend) — add `ActParams QuantizationParams` field to `QuantizedDense[T]`; `SetActParams(p QuantizationParams)` setter called by `Quantize[T]` when `cfg.Mode == FullInt8`; full-int8 `Forward` branch: quantize input `x_int8[k] = quantize(float64(x[k]), ActParams, 0)` per element; int32 accumulator GEMM `y_int32[c] = Σ_k int32(W[c,k]) * int32(x_int8[k])`; per-channel scale correction `y_float[c] = scale_W[c] * scale_x * float64(y_int32[c]) + Bias[c]`; branch selected by `ActParams.Scale != nil` (non-nil = full-int8 mode)
- [ ] [T-19A09] `pkg/quantization/evaluator.go` — `EvalSample[T]` struct (`Input []T`, `Expected []T`); `EvaluationResult` struct (`MetricFloat float64`, `MetricQuant float64`, `DeltaRelative float64`, `PerLayerL2 []float64`); `Evaluate[T](baseline *nn.Network[T], quantized *QuantizedNetwork[T], evalSet []EvalSample[T], metricFn func(output, expected []T) float64) (EvaluationResult, error)` — runs both networks on every eval sample, computes metricFn, per-layer L2 norm delta; no auto-accept gate (QUANT-C6 — informational only); `PerLayerL2[i] = ||y_float_i − y_quant_i||₂ / ||y_float_i||₂`

### Track A — Phase γ: Attention Projection Quantization (A10 → A11)

- [ ] [T-19A10] `pkg/quantization/attn.go` — `QuantizedAttentionProjections[T]` struct (`WqParams, WkParams, WvParams, WoParams QuantizationParams`, `Wq, Wk, Wv, Wo []int8`, `Dmodel, NumHeads int`); weight-only `Forward(x []T) []T`: dequantizes each of Wq/Wk/Wv/Wo on the fly, executes the standard multi-head attention forward (scaled dot-product + softmax) using float arithmetic for scoring and softmax (QUANT-10 — only projection weights quantized); constructor `newQuantizedAttentionProjections[T]` reading `Wq/Wk/Wv/Wo` public fields from `*attention.MultiHeadAttention[T]`
- [ ] [T-19A11] `pkg/quantization/quantizer.go` (amend) — `Quantize[T]` type-switch gains `case *attention.MultiHeadAttention[T]`: calls `newQuantizedAttentionProjections[T]` to quantize the four projection matrices and adds the result to `QuantizedNetwork[T].Layers`; `QuantizedLayerJSON.Type = "AttentionProj"` for persistence; `UnmarshalQNN` gains corresponding case to reconstruct `QuantizedAttentionProjections[T]` from JSON

### Validation

- [ ] [T-19T01] Golden-value + round-trip tests — `TestQuantizationGoldenValue`: `computeSymmetric(0.0, 3.0)` → `scale≈0.0236, zero_point=0`; `quantize(1.0, p, 0)` → `42` (roundHalfEven L1 §4.2); `dequantize(42, p, 0)` ≈ `0.9912` (≤ 0.01 abs error); `TestRoundHalfEven`: check 0.5→0, 1.5→2, 2.5→2, 3.5→4 (banker's rounding table from L1 §4.2); `TestQNNRoundTrip`: `MarshalQNN(Quantize(net))` → `UnmarshalQNN` → `Forward(x)` matches pre-marshal bit-exact; per-module unit tests for params/config/dense/conv/persistence; `go test ./pkg/quantization/... -cover` PASS; coverage ≥ 80%
- [ ] [T-19T02] Full-int8 accuracy gate + attention quantization — `TestFullInt8Forward`: 64-neuron `QuantizedDense[T]` with `CalibrationRunner` (100 random samples): `||y_float − y_quant||_∞ ≤ 5%·||y_float||_∞` per QUANT-5 verify criterion; `TestCalibrationRunner_TooFewSamples`: 10 samples → `ErrCalibTooFewSamples`; `TestDegenerateRange`: constant-activation layer → `ErrDegenerateRange` + fallback `scale=1.0`; `TestQuantizedAttentionForward`: `Dmodel=64, NumHeads=4, SeqLen=8` — FD-equivalent `||delta||_∞ ≤ 5%`; `TestSideBySideEvaluator`: reports non-zero `PerLayerL2`; `go test -race ./pkg/quantization/...` clean

### Gate

- [ ] [T-19Z01] Phase 19 release gate — `go build ./...` clean (default tags, no cgo); `go test ./pkg/...` green; coverage floors: `pkg/quantization/` ≥ 80% (C30); CHANGELOG.md `[0.17.0]` entry written with "Post-Training Quantization" bullet list (weight-only + full-int8 + attention projections + evaluator); `v0.17.0` tag prepared (user runs `git tag -a v0.17.0` — agent never auto-tags); frontmatter `provides` lists `Quantize[T]`, `QuantizedNetwork[T]`, `CalibrationRunner[T]`, `Evaluate[T]` availability; `Load[T]` documented in CHANGELOG

## Detailed Tracking

### [T-19A01] `pkg/quantization/config.go` + `doc.go`

- **Spec:** `l2-quantization-impl.md` §4.2 (`QuantMode`, `QuantizationConfig[T]`)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go build ./pkg/quantization/...` clean; `DefaultQuantizationConfig[T]()` returns `{Mode:WeightOnly, WeightGranularity:PerChannel, Strategy:Percentile99p9, PercentileThreshold:99.9, MinCalibSamples:32}`
- **Handoff:** A02 imports `QuantizationConfig[T]`; A03 builds `QuantizedNetwork[T]` with it.
- **Notes:** `QuantMode` and `Granularity`/`CalibStrategy` are in separate files (config.go vs params.go) per the spec §4.1 layout. `CalibrationProvenance` struct includes `SampleCount int`, `Seed uint64`, `Strategy CalibStrategy` — used by `MarshalQNN` for `.qnn.json` traceability. AI-Meta block: `// @ai:stability Experimental`.

### [T-19A02] `pkg/quantization/params.go`

- **Spec:** `l2-quantization-impl.md` §3 QUANT-1/QUANT-2 + §6.1 (roundHalfEven)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run 'TestRoundHalfEven|TestQuantizationGoldenValue|TestComputeSymmetric' ./pkg/quantization/` PASS — banker's rounding table + L1 §4.2 golden values; `computeSymmetric(0.0, 3.0).Scale[0]` ≈ 0.023622047 (≤1e-6 abs err); `quantize(1.0, p, 0)` = 42; `dequantize(42, p, 0)` ≈ 0.9921 (≤0.01 abs err)
- **Handoff:** A03/A04/A05/A07 all depend on the pure functions here; T-19T01 is the normative gate.
- **Notes:** `roundHalfEven(x float64) int64` — the standard library `math.Round` uses round-half-away-from-zero, NOT banker's rounding. Implement as: if `x - math.Floor(x) == 0.5` and `int64(math.Floor(x)) % 2 == 0` → round down, else `math.Round(x)`. The L1 §4.2 table is authoritative: `round(0.5)=0, round(1.5)=2, round(2.5)=2, round(3.5)=4`. `computeSymmetric` reserves the `-128` slot (range `[-127, 127]`), so `scale = rMax / 127.0` when `rMax > -rMin`. `computeAsymmetric` uses full `[-128, 127]` range: `scale = (rMax - rMin) / 255.0`; `zp = int32(math.Round(-rMin / scale)) - 128`.

### [T-19A03] `pkg/quantization/quantizer.go`

- **Spec:** `l2-quantization-impl.md` §4.2 (`QuantizedNetwork[T]`, `Quantize[T]`) + §6.5 (ConvPrefix access)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestQuantize ./pkg/quantization/` PASS — `Quantize[T]` on a 2-layer Dense network returns a `QuantizedNetwork[T]` with `len(Layers)==2`; source network's weights are unchanged; `QuantizedNetwork[T].Forward(x)` returns a `[]T` of the same length as the network's output
- **Handoff:** A04/A05 provide the type-switch cases; A06 adds persistence; T-19T01 verifies Forward output.
- **Notes:** `Quantize[T]` iterates `net.Config().ConvPrefix` (type `[]layer.Layer[T]`) in order. Each element's Go type is revealed via type-switch — this is the same pattern as `applyConvBackward` in `pkg/nn/train.go` (canonical reference `[NN-TRAIN]`). Unknown layer types (e.g. `*norm.LayerNorm[T]`) pass through as a float identity wrapper (`floatPassthrough[T]`) per QUANT-10. `BaselineHash` = `hex.EncodeToString(sha256.Sum256(jsonBytes)[:])` where `jsonBytes` is the canonical JSON of the source network (not file path). `calibSamples == nil` is valid for WeightOnly mode — guard this path explicitly.

### [T-19A04] `pkg/quantization/dense.go`

- **Spec:** `l2-quantization-impl.md` §4.3 (weight-only forward path) + §3 QUANT-5
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run 'TestQuantizedDense' ./pkg/quantization/` PASS — `QuantizedDense.Forward` output matches `float Dense.Forward` within `5%·||y_float||_∞` tolerance on a `Cout=16, Cin=8` random dense with `seed=42`; `TestQuantizedDense_WeightShape`: `len(Weights) == Cout*Cin`; `len(WeightParams.Scale) == Cout` for PerChannel
- **Handoff:** T-19A08 amends this file to add the full-int8 branch.
- **Notes:** Weight layout is row-major `Cout × Cin` (matches `layer.Dense[T]` layout per `[LAYER-DENSE]`). The weight-only forward reconstructs each row of `W` from int8 + scale + zero_point at inference time — O(Cout × Cin) float multiplications, same FLOP count as float GEMM but 4× smaller in cache. Access `Dense[T].Weights []T` and `Dense[T].Bias []T` directly via struct field access (same module, GC-3 read-only).

### [T-19A05] `pkg/quantization/conv.go`

- **Spec:** `l2-quantization-impl.md` §4.1 (`QuantizedConv1D[T]`, `QuantizedConv2D[T]`)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run 'TestQuantizedConv' ./pkg/quantization/` PASS — Conv1D + Conv2D forward output matches float within `5%·||y_float||_∞` on random small filter (`Cout=4, Cin=2, Kernel=3`); `TestQuantizedConv_WeightShape` assertions on int8 buffer sizes
- **Handoff:** A06 serializes Conv1D/Conv2D layers in `QuantizedLayerJSON.Type` = "Conv1D" / "Conv2D".
- **Notes:** Conv1D filter layout: `[OutChannels × InChannels × KernelSize]` row-major (verify against `pkg/layer/conv/conv1d.go` `[LAYER-CONV]`). PerChannel granularity groups along `OutChannels` axis (Cout = OutChannels). Conv2D filter layout: `[OutChannels × InChannels × KH × KW]`. The weight-only forward dequantizes filters on the fly per output channel before the spatial convolution loop — no kernel fusion in v0.1 (QUANT-9).

### [T-19A06] `pkg/quantization/persistence.go`

- **Spec:** `l2-quantization-impl.md` §4.6 (`.qnn.json` schema) + §3 QUANT-6
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run 'TestQNNRoundTrip|TestQNNWrongType' ./pkg/quantization/` PASS — `MarshalQNN` + `UnmarshalQNN` round-trip preserves all int8 weight bytes bit-exact; decoded `Scale` arrays match float64 bit-exactly; wrong `"Type"` tag → error; `TestQNNSave`: writes to temp file, `Load[T]` reads it back, `Forward` output matches pre-save
- **Handoff:** T-19Z01 gate verifies the `.qnn.json` artifact is human-readable + self-contained.
- **Notes:** `"Type":"quantized"` is the top-level discriminator (analogous to `.nn.json` type tag in `l2-persistence-impl.md`). `Weights` field: `base64.StdEncoding.EncodeToString(rawInt8Bytes)` where `rawInt8Bytes` is the `[]int8` cast to `[]byte` via a plain copy loop (no `unsafe`). `Bias` field: encode `[]float64` as IEEE 754 little-endian `[]byte` (8 bytes per float64) then base64 — NOT int32, despite the spec §4.6 draft saying int32; bias stays float64 for full dynamic range (spec note §7 alternative float32 rejected). Round-trip invariant: integer weight bytes survive with zero float intermediates; float64 bias bytes survive with zero rounding.

### [T-19A07] `pkg/quantization/calibration.go`

- **Spec:** `l2-quantization-impl.md` §4.5 + §3 QUANT-4/QUANT-7
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run 'TestCalibrationRunner' ./pkg/quantization/` PASS — `TestCalibrationRunner_TooFewSamples`: 10 samples → `ErrCalibTooFewSamples`; `TestCalibrationRunner_MinMax`: 50 samples uniform `[−1, 1]` → `Params().Scale[0] ≈ 1.0/127.0`; `TestCalibrationRunner_Percentile99p9`: 200 samples with 3 outliers at ±100 → effective max < 2.0 (outliers clipped); `TestDegenerateRange`: all-zero activations → `ErrDegenerateRange` + `scale=1.0, zp=0`
- **Handoff:** A08 calls `SetActParams` with the activation params returned by `Params()`.
- **Notes:** `Run(samples)` must drive the source network's `Forward` method one sample at a time (or batched if network supports it). Activation statistics are collected between layers using a hook or by reading intermediate outputs — in v0.1 the simplest approach is to run `Quantize[T]` with `calibSamples != nil`, pass samples through each quantized layer's float pass-through wrapper, and record min/max of each layer's input. The histogram for `Percentile99p9` uses 512 uniform buckets over `[min, max]` (adequate for 99.9th-percentile clipping at 200+ samples). Add `ErrCalibTooFewSamples` and `ErrDegenerateRange` to `pkg/utils/errors.go` under the `QUANT` category (C32).

### [T-19A08] `pkg/quantization/dense.go` (amend — full-int8)

- **Spec:** `l2-quantization-impl.md` §4.4 + §3 QUANT-5 (full-int8 path)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run 'TestFullInt8Forward' ./pkg/quantization/` PASS — 64-neuron `QuantizedDense[T]` calibrated with 100 random samples: `||y_float − y_quant||_∞ ≤ 5%·||y_float||_∞`; `TestFullInt8VsWeightOnly`: full-int8 output is within 2× the weight-only error (not necessarily smaller, but in the same ballpark)
- **Handoff:** T-19T02 is the normative accuracy gate for the full-int8 path.
- **Notes:** `bias_correction[c]` pre-computed at quantization time: `bc[c] = −scale_W[c] * zp_W[c] * Σ_k x_int8[k]` — but since we don't know `x_int8` at quantization time, compute it at inference time as part of the accumulator: `y_int32[c] += −zp_W[c] * Σ_k int32(x_int8[k])`. The full-int8 branch is activated by `ActParams.Scale != nil` (set by `Quantize[T]` when `cfg.Mode == FullInt8` and calibration was run). WeightOnly path (`ActParams.Scale == nil`) unchanged.

### [T-19A09] `pkg/quantization/evaluator.go`

- **Spec:** `l2-quantization-impl.md` §4.7 + §3 QUANT-8
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run 'TestSideBySideEvaluator' ./pkg/quantization/` PASS — `DeltaRelative` is non-zero; `PerLayerL2` has `len == len(evalSet layers)`; `metricFn` is called once per sample for both networks; `ErrEmptyEvalSet` returned for empty `evalSet`
- **Handoff:** T-19Z01 gate: CHANGELOG notes the evaluator as informational (no auto-accept gate, QUANT-C6).
- **Notes:** `PerLayerL2[i] = ||y_float_i − y_quant_i||₂ / ||y_float_i||₂` where index `i` is the layer index in `QuantizedNetwork[T].Layers`. For layers where `||y_float_i||₂ < 1e-10`, report 0 to avoid division-by-zero. `DeltaRelative = (MetricQuant − MetricFloat) / |MetricFloat|`; if `MetricFloat == 0`, report 0.

### [T-19A10] `pkg/quantization/attn.go`

- **Spec:** `l2-quantization-impl.md` §5 Phase γ + §3 QUANT-10 (attention projections Phase γ)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run 'TestQuantizedAttentionForward' ./pkg/quantization/` PASS — `Dmodel=64, NumHeads=4, SeqLen=8`: `||y_float − y_quant||_∞ ≤ 5%·||y_float||_∞`; `TestQuantizedAttnWeightShape`: `len(Wq) == Dmodel*Dmodel` (same for Wk/Wv/Wo); causal masking preserved (output for non-causal vs causal differs as expected)
- **Handoff:** A11 wires this into `Quantize[T]` type-switch.
- **Notes:** **Phase γ is gated on Phase 18 closeout** — `*attention.MultiHeadAttention[T]` must exist and its `Wq/Wk/Wv/Wo` fields must be accessible (they are public in `pkg/layer/attention/multihead.go`). The scoring matrix `Q·Kᵀ/√d_k` and softmax remain float (QUANT-10). `QuantizedAttentionProjections[T].Forward` reconstructs each projection matrix from int8 + scale on the fly, then executes the standard scaled-dot-product attention in float — identical to `MultiHeadAttention[T].Forward` except the four GEMM calls use dequantized weight rows.

### [T-19A11] `pkg/quantization/quantizer.go` (amend — MHA type-switch)

- **Spec:** `l2-quantization-impl.md` §5 Phase γ (step 12)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run 'TestQuantizeMHA' ./pkg/quantization/` PASS — `Quantize[T]` on a network whose `ConvPrefix` contains a `*attention.MultiHeadAttention[T]` produces a `QuantizedNetwork[T]` with a `QuantizedAttentionProjections[T]` layer at the corresponding index; round-trip via `MarshalQNN`/`UnmarshalQNN` restores the attention layer type
- **Handoff:** Closes Phase γ; T-19T02 verifies attention quantization accuracy.
- **Notes:** The type-switch case is `case *attention.MultiHeadAttention[T]: return newQuantizedAttentionProjections[T](l, cfg)`. `QuantizedLayerJSON.Type = "AttentionProj"` for persistence. `UnmarshalQNN` gains a corresponding `case "AttentionProj"` branch. Import `pkg/layer/attention` in `quantizer.go` — this is the only cross-package import added in Phase γ; it is a same-module peer dependency.

### [T-19T01] Validation — golden-value + round-trip tests

- **Goal:** Verify QUANT-1/QUANT-2 (affine math) and QUANT-6 (persistence) against normative L1 §4.2 values.
- **Method:** `go test -run 'TestRoundHalfEven|TestQuantizationGoldenValue|TestQNNRoundTrip|TestQNNSave' -count=1 ./pkg/quantization/`. Normative anchor: `computeSymmetric(0.0, 3.0).Scale[0] ≈ 3.0/127 ≈ 0.023622047`; `quantize(1.0, p, 0) = 42`; `dequantize(42, p, 0) ≈ 0.9921`. Coverage: `go test ./pkg/quantization/... -cover` ≥ 80%.
- **Status:** Todo
- **Notes:** The L1 §4.2 example is normative — if `quantize(1.0, p, 0) ≠ 42`, the `roundHalfEven` implementation is wrong. Check: `1.0 / (3.0/127) = 42.333...`; `roundHalfEven(42.333) = 42`. Also check the boundary case `quantize(1.5, p, 0)`: `1.5 / (3.0/127) = 63.5`; `roundHalfEven(63.5) = 64` (nearest even). The round-trip test must use a real network (not a stub) to exercise the full `Quantize → MarshalQNN → UnmarshalQNN → Forward` pipeline.

### [T-19T02] Validation — full-int8 accuracy + attention quantization gate

- **Goal:** Verify QUANT-5 (full-int8 ≤ 5% error) and Phase γ attention quantization accuracy.
- **Method:** `go test -race -run 'TestFullInt8Forward|TestCalibrationRunner|TestDegenerateRange|TestQuantizedAttentionForward|TestSideBySideEvaluator' -count=1 ./pkg/quantization/`. `go test -race ./pkg/quantization/...` clean.
- **Status:** Todo
- **Notes:** `TestFullInt8Forward` uses a fixed `rand.New(rand.NewPCG(42, 42))` seed for reproducibility. The 5% tolerance is for `||·||_∞` (max-absolute), not average error — a single large outlier can fail the gate. If the gate fails, the most likely cause is incorrect `bias_correction` or wrong `zp_W` subtraction in the int32 accumulator.

### [T-19Z01] Phase 19 release gate

- **Goal:** Confirm Phase 19 v0.17.0 RC ready.
- **Method:** `go build ./...` clean (default tags, no cgo); `go test ./pkg/...` green (skipping `-race` on Windows per documented CGO limitation); coverage floor: `pkg/quantization/` ≥ 80% (C30); CHANGELOG.md `[0.17.0]` entry written with "Post-Training Quantization (PTQ)" bullet list covering: `Quantize[T]` + weight-only Dense/Conv1D/Conv2D, `CalibrationRunner[T]` + full-int8 path, `QuantizedAttentionProjections[T]`, `Evaluate[T]` side-by-side evaluator, `.qnn.json` persistence; `v0.17.0` tag prepared (user runs `git tag -a v0.17.0` — agent never auto-tags); frontmatter `provides` updated; `Load[T]` documented.
- **Status:** Todo
- **Notes:** Phase 20+ (NLP examples E17+, BERT-Base end-to-end, SIMD backend kernels) remains deferred. The `pkg/compute/cpu/` SIMD extension point is documented in `quantizer.go` AI-Meta but not wired in v0.1 (QUANT-9). v0.17.0 is the first release where users can quantize a trained GoNN network to int8 and run inference from a `.qnn.json` file without the training environment.
