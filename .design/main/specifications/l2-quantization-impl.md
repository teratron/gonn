# Quantization Implementation

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-quantization.md

## Overview

Go realization of [l1-quantization.md](l1-quantization.md) — post-training int8 quantization delivered as the `pkg/quantization/` package. Consumes a trained `*nn.Network[T]` plus an optional calibration set; produces a deployable `QuantizedNetwork[T]` artifact persisted as `.qnn.json`.

Three exported top-level operations: `Quantize[T]` (transforms a float network into a quantized one), `Load[T]` (loads a `.qnn.json` artifact), and `Evaluate[T]` (side-by-side float vs quant comparison per QUANT-8). Quantized layers own int8 weight arrays alongside float `(scale, zero_point)` metadata; their `Forward` dequantizes on the fly in weight-only mode or performs an int32-accumulator GEMM in full-int8 mode. No changes to `pkg/nn/`, `pkg/layer/`, or `pkg/compute/` are required in v0.1 — quantization is a peer transformation layer, not an extension of the training-time graph.

Coverage target ≥ 80% per C30.

## Related Specifications

- [l1-quantization.md](l1-quantization.md) — L1 contract: affine scheme, per-channel granularity, calibration protocol, `.qnn.json` schema
- [l2-persistence-impl.md](l2-persistence-impl.md) — Sibling `.qnn.json` artifact shares type-discriminator pattern and round-trip invariants with `.nn.json`
- [l2-nn-facade.md](l2-nn-facade.md) — `*nn.Network[T]` is the input to `Quantize`; no new facade options needed in v0.1
- [l2-layer-types.md](l2-layer-types.md) — `layer.Layer[T]` types (Dense, Conv1D, Conv2D) are the quantization targets in v0.1
- [l2-normalization-impl.md](l2-normalization-impl.md) — LayerNorm/BatchNorm excluded from quantization in v0.1 (QUANT-10)
- [l2-transformer-impl.md](l2-transformer-impl.md) — Attention projection matrices (Wq/Wk/Wv/Wo) are Phase γ targets; scoring / softmax stay float per QUANT-10
- [l2-errors-impl.md](l2-errors-impl.md) — Calibration errors (insufficient samples, degenerate range) follow C32 error taxonomy

## 1. Motivation

Phase 18 lands `EncoderBlock[T]` / `DecoderBlock[T]` / `Stack[T]`. A BERT-Base-style 12-layer stack has ~85M parameters consuming ~340 MB at float32. The `l1-quantization.md` L1 contract is already Stable — this L2 spec closes the gap between a defined math contract and a runnable Go implementation.

Three user goals this spec enables:

1. **Storage reduction**: float32 model → int8 weights = 4× compression; a 340 MB BERT-Base drops to ~85 MB.
2. **Inference speed**: int8 GEMM on AVX-512 VNNI is 2–4× faster; even without SIMD, memory traffic reduction alone improves latency.
3. **Portable artifacts**: `.qnn.json` is a self-contained inference artifact loadable without the training environment.

## 2. Constraints & Assumptions

All QUANT-C1..QUANT-C8 from the L1 spec apply verbatim. Go-specific additions:

- **GC-1 (No new external deps)**: integer arithmetic uses only `math/bits` and standard library. No BLAS, no CGO, no SIMD intrinsics in v0.1 reference implementation. SIMD kernels are a `pkg/compute/cpu/` backend extension (QUANT-9) reserved for a separate pass.
- **GC-2 (`utils.Float` preserved)**: `QuantizedNetwork[T]` keeps its generic `T` parameter for the dequantize-to-float output path. int8 weight arrays are `[]int8` — unparameterized by `T` — alongside float scale arrays typed as `[]float64` (not `[]T`), because calibration statistics are always high-precision.
- **GC-3 (Read-only network input)**: `Quantize[T](net *nn.Network[T], ...)` does NOT mutate the source network. It reads weights via the existing `layer.Layer[T]` interface using `GradSlots` / direct struct access via layer type-switch. The source network remains trainable after quantization.
- **GC-4 (Phase-gated scope)**: Phase α implements weight-only quantization for Dense, Conv1D, Conv2D. Phase β adds activation calibration + full-int8 path. Phase γ adds attention projection quantization. The `pkg/quantization/` package is callable after Phase α; later phases extend it additively.
- **GC-5 (No `layer.Layer[T]` for quantized layers)**: `QuantizedDense[T]` and peers do NOT implement `layer.Layer[T]` — they are inference-only types and have no `Backward`. A `QuantizedNetwork[T]` exposes `Forward([]T) []T` directly; it is NOT wirable into `pkg/nn/` ConvPrefix slot.

## 3. Invariant Compliance

| L1 Invariant | Go Implementation |
| :--- | :--- |
| **QUANT-1** (affine reconstruction `r = scale·(q − zero_point)`) | `dequantize(q int8, p QuantizationParams, ch int) float64` inline function in `params.go`; `quantize(r float64, p QuantizationParams, ch int) int8` uses `math.Round` (banker's rounding via `roundHalfEven`). Both are pure functions tested against L1 §4.2 example values. |
| **QUANT-2** (symmetric vs asymmetric range mapping) | `computeSymmetric(rMin, rMax float64) QuantizationParams` and `computeAsymmetric(rMin, rMax float64) QuantizationParams` in `params.go`. Symmetric reserves `−128` slot (range `[−127,127]`). Test: per L1 §4.2 example — `r_max=3.0 → scale≈0.0236, zero_point=0, q(1.0)=42`. |
| **QUANT-3** (per-channel along Cout for weights) | `quantizeWeightsTensor(w []float64, cout, cin int, gran Granularity) ([]int8, QuantizationParams)` in `quantizer.go`; `PerChannel` branches on `Cout` rows; `PerTensor` treats entire `w` as one range. Assert: `len(params.Scale) == Cout` for `PerChannel`, `len==1` for `PerTensor`. |
| **QUANT-4** (calibration protocol; 32-hard / 100-soft sample floors) | `CalibrationRunner[T].Run(samples [][]T)` in `calibration.go`; counts samples, returns `ErrCalibTooFewSamples` for `n < 32`, logs warning for `n < 100`. Strategy dispatch: `MinMax`, `Percentile99p9` (both mandatory), `Entropy` (optional, Phase β+). |
| **QUANT-5** (quantized Forward; int32 accumulator) | `QuantizedDense[T].Forward(x []T) []T`: weight-only path dequantizes weights → float32 on the fly then does float GEMM. Full-int8 path (Phase β): int32 accumulator GEMM + per-output-channel scale correction per L1 §4.4. Verify: `||y_float − y_quant||_∞ ≤ 5%·||y_float||_∞` asserted in test via `SideBySideEvaluator`. |
| **QUANT-6** (`.qnn.json` sibling artifact) | `persistence.go`: `MarshalQNN[T]` / `UnmarshalQNN[T]`; top-level `"Type":"quantized"` discriminator; `BaselineHash` SHA-256 of source `.nn.json`; `CalibrationProvenance {SampleCount, Seed, Strategy}`; per-layer `Scale[]`, `ZeroPoint[]`, `Weights` base64-encoded int8, `Bias` base64-encoded int32. Round-trip preserves all integer bits. |
| **QUANT-7** (calibration sample diversity; min/max collapse guard) | `CalibrationRunner.Run` detects `r_max − r_min < 1e-6`; logs `ErrDegenerateRange` + falls back to `scale=1.0, zero_point=0` for that layer. Per-layer tracking. |
| **QUANT-8** (side-by-side evaluation helper) | `evaluator.go`: `Evaluate[T](baseline, quantized Network, evalSet, metricFn)` returns `EvaluationResult{MetricFloat, MetricQuant, DeltaRelative, PerLayerL2}`. No auto gate — informational only (QUANT-C6). |
| **QUANT-9** (backend dispatch boundary) | v0.1 reference implementation uses pure Go (no SIMD, no CGO). A `pkg/compute/cpu/` kernel extension point (`Int8GEMM` kernel registration) is documented in `quantizer.go` AI-Meta but not wired in Phase α–γ. Platform-specific VNNI paths are a separate future PR. |
| **QUANT-10** (layer compatibility coverage) | Phase α: Dense + Conv1D + Conv2D weight-only. Phase γ: attention projection matrices (Wq/Wk/Wv/Wo) via `QuantizedAttentionProjections[T]`. Out-of-scope in v0.1: LayerNorm (γ/β), recurrent weights, embedding tables (`W_E`). Enforced via `quantizableLayer[T]` type-switch — unrecognized layer types pass through as float identity. |

## 4. Detailed Design

### 4.1 Package Layout

```plaintext
pkg/quantization/
├── doc.go           // package doc + AI-Meta; import aliases
├── config.go        // QuantizationConfig[T], QuantMode, LayerQuantConfig, enums
├── params.go        // QuantizationParams, Granularity, CalibStrategy, QuantDtype
│                    // dequantize(), quantize(), roundHalfEven(), computeSymmetric/Asymmetric
├── quantizer.go     // Quantize[T]() entry; QuantizedNetwork[T]; layer type-switch dispatch
├── dense.go         // QuantizedDense[T].Forward() (weight-only Phase α; full-int8 Phase β)
├── conv.go          // QuantizedConv1D[T], QuantizedConv2D[T].Forward() (Phase α)
├── attn.go          // QuantizedAttentionProjections[T].Forward() (Phase γ)
├── calibration.go   // CalibrationRunner[T]; MinMax, Percentile99p9 strategies
├── persistence.go   // MarshalQNN[T] / UnmarshalQNN[T]; .qnn.json schema
├── evaluator.go     // SideBySideEvaluator[T]; EvaluationResult
└── *_test.go
```

### 4.2 Core Types

```go
// [REFERENCE] pkg/quantization/config.go

type QuantMode int
const (
    WeightOnly QuantMode = iota // int8 weights, float activations (Phase α)
    FullInt8                    // int8 weights + int8 activations, int32 accumulator (Phase β+)
)

type QuantizationConfig[T utils.Float] struct {
    Mode                QuantMode
    WeightGranularity   Granularity    // PerTensor | PerChannel (default PerChannel)
    ActGranularity      Granularity    // PerTensor only in v0.1
    Strategy            CalibStrategy  // MinMax | Percentile99p9 (default)
    PercentileThreshold float64        // default 99.9; overridable
    MinCalibSamples     int            // hard floor 32; soft floor 100
    Seed                uint64         // for calibration provenance hash
}
```

```go
// [REFERENCE] pkg/quantization/params.go

type Granularity int
const (
    PerTensor  Granularity = iota
    PerChannel             // weights only (QUANT-C4)
)

type CalibStrategy int
const (
    MinMax         CalibStrategy = iota
    Percentile99p9               // default
    Entropy                      // optional, Phase β+
)

type QuantizationParams struct {
    Scale      []float64  // len=1 per-tensor; len=Cout per-channel
    ZeroPoint  []int32    // same len as Scale
    Granularity Granularity
    Strategy    CalibStrategy
}
```

```go
// [REFERENCE] pkg/quantization/quantizer.go

type QuantizedNetwork[T utils.Float] struct {
    Config           QuantizationConfig[T]
    Layers           []quantizedLayer[T]  // private interface
    BaselineHash     string               // SHA-256 of source .nn.json
    CalibProvenance  CalibrationProvenance
}

// Quantize transforms a trained network. Does NOT mutate net.
func Quantize[T utils.Float](
    net *nn.Network[T],
    calibSamples [][]T,      // nil = weight-only (no activation calibration)
    cfg QuantizationConfig[T],
) (*QuantizedNetwork[T], error)

// Load deserialises a .qnn.json file.
func Load[T utils.Float](path string) (*QuantizedNetwork[T], error)

// Evaluate runs side-by-side comparison (QUANT-8).
func Evaluate[T utils.Float](
    baseline    *nn.Network[T],
    quantized   *QuantizedNetwork[T],
    evalSet     []EvalSample[T],
    metricFn    func(output, expected []T) float64,
) (EvaluationResult, error)
```

### 4.3 Weight-Only Forward Path (Phase α)

For a Dense layer with weight matrix `W ∈ ℝ^{Cout × Cin}`:

1. At quantization time: `quantizeWeightsTensor(W, Cout, Cin, PerChannel)` → `W_int8 []int8` + `WeightParams`.
2. At inference time (`QuantizedDense[T].Forward(x []T)`):
   - For each output channel `c ∈ [0, Cout)`: reconstruct `w_c = scale[c] · (W_int8_row[c] − zero_point[c])` as `[]float64`, dot-product with `x`, add bias.
   - Cost: O(Cout × Cin) float multiplications — same as float path, but `W_int8` is 4× smaller in cache.
3. Activation passes through as `T` (no quantization of activations in WeightOnly mode).

This is the baseline correctness path. SIMD-accelerated int8 GEMM (AVX-512 VNNI) is a backend extension that replaces step 2 with an integer accumulator path without changing the API.

### 4.4 Full-Int8 Forward Path (Phase β)

Adds `ActParams QuantizationParams` to each `QuantizedDense[T]` filled by `CalibrationRunner`:

1. Quantize input: `x_int8 = quantize(x_float, ActParams)` per element.
2. int32 accumulator GEMM: `y_int32[c] = Σ_k W_int8[c,k] · x_int8[k]` (pure Go, no SIMD).
3. Apply per-channel scale correction + dequantize output:
   ```
   y_float[c] = scale_W[c] · scale_x · y_int32[c] + bias_correction[c]
   ```
4. Optional re-quantize to int8 for next layer (QUANT-C7 FullInt8 pipeline).

`bias_correction[c]` is the zero-point correction term pre-computed at quantization time (L1 §4.4), stored alongside `W_int8` to keep the inference loop to a single GEMM plus one add per output channel.

### 4.5 Calibration Runner

```go
// [REFERENCE] pkg/quantization/calibration.go

type CalibrationRunner[T utils.Float] struct {
    net     *nn.Network[T]
    cfg     QuantizationConfig[T]
    stats   map[int]layerStats  // keyed by ConvPrefix index
}

type layerStats struct {
    min, max float64
    histogram []int64  // for entropy strategy
    count     int
}

// Run passes all calibration samples through the network,
// recording per-layer activation statistics.
func (c *CalibrationRunner[T]) Run(samples [][]T) error

// Params returns the derived QuantizationParams for each layer.
func (c *CalibrationRunner[T]) Params() (map[int]QuantizationParams, error)
```

Strategy dispatch in `Params()`:
- `MinMax`: `scale, zp = computeSymmetric(stats.min, stats.max)` or asymmetric depending on sign of min.
- `Percentile99p9`: collects the histogram during `Run`, uses the 99.9th percentile of `|A|` as effective max, then applies the same `computeSymmetric / computeAsymmetric` with clipped range.
- Degenerate guard: `max − min < 1e-6` → fall back to `scale=1.0, zp=0`, wrap `ErrDegenerateRange`.

### 4.6 Persistence Schema

```go
// [REFERENCE] pkg/quantization/persistence.go

// QuantizedNetworkJSON is the wire format for .qnn.json.
type QuantizedNetworkJSON struct {
    Type                string                  `json:"Type"`   // "quantized"
    Version             string                  `json:"Version"` // "0.1.0"
    BaselineHash        string                  `json:"BaselineHash"`
    CalibrationProvenance CalibrationProvenanceJSON `json:"CalibrationProvenance"`
    Mode                string                  `json:"Mode"`   // "weight_only" | "full_int8"
    Layers              []QuantizedLayerJSON    `json:"Layers"`
}

type QuantizedLayerJSON struct {
    Index       int       `json:"Index"`
    Type        string    `json:"Type"`        // "Dense" | "Conv1D" | "Conv2D" | "AttentionProj"
    Cin         int       `json:"Cin"`
    Cout        int       `json:"Cout"`
    Granularity string    `json:"Granularity"` // "per_tensor" | "per_channel"
    Strategy    string    `json:"Strategy"`
    Scale       []float64 `json:"Scale"`
    ZeroPoint   []int32   `json:"ZeroPoint"`
    Weights     string    `json:"Weights"`  // base64(int8 raw bytes, row-major Cout×Cin)
    Bias        string    `json:"Bias"`     // base64(int32 raw bytes, len=Cout)
}
```

Round-trip invariant: `UnmarshalQNN(MarshalQNN(qnet))` produces identical int8 weight bytes and float64 scale arrays (no float intermediate for integer fields).

### 4.7 Side-by-Side Evaluator

```go
// [REFERENCE] pkg/quantization/evaluator.go

type EvalSample[T utils.Float] struct {
    Input    []T
    Expected []T
}

type EvaluationResult struct {
    MetricFloat   float64
    MetricQuant   float64
    DeltaRelative float64            // (MetricQuant − MetricFloat) / |MetricFloat|
    PerLayerL2    []float64          // ||y_float − y_quant||₂ / ||y_float||₂ per layer
}

func Evaluate[T utils.Float](
    baseline  *nn.Network[T],
    quantized *QuantizedNetwork[T],
    evalSet   []EvalSample[T],
    metricFn  func(output, expected []T) float64,
) (EvaluationResult, error)
```

No automatic accept/reject gate (QUANT-C6). `DeltaRelative` is negative when quantization improves the metric (unlikely but possible due to regularization effect).

## 5. Implementation Plan (Three-Phase)

### Phase α — Weight-Only Int8 (Dense, Conv1D, Conv2D)

Minimal deployable subset. No calibration runner needed — weight stats are computed from the frozen weight tensor.

1. `pkg/quantization/config.go` — enums + `QuantizationConfig[T]`
2. `pkg/quantization/params.go` — `QuantizationParams` + affine math helpers
3. `pkg/quantization/quantizer.go` — `Quantize[T]` type-switch dispatcher + `QuantizedNetwork[T]`
4. `pkg/quantization/dense.go` — `QuantizedDense[T]` weight-only Forward
5. `pkg/quantization/conv.go` — `QuantizedConv1D[T]`, `QuantizedConv2D[T]` weight-only Forward
6. `pkg/quantization/persistence.go` — `MarshalQNN` / `UnmarshalQNN`
7. `*_test.go` — per-module unit tests + golden-value regression (L1 §4.2 example)

**Verify criterion (Phase α gate)**: `go test ./pkg/quantization/... -cover` PASS; coverage ≥ 80%; golden-value test matches L1 §4.2 example exactly (`scale≈0.0236, zero_point=0, q(1.0)=42`); `.qnn.json` round-trip restores all int8 weight bytes bit-exact.

### Phase β — Activation Calibration + Full-Int8 Path

8. `pkg/quantization/calibration.go` — `CalibrationRunner[T]`; MinMax + Percentile99p9 strategies
9. `pkg/quantization/dense.go` (amend) — full-int8 Forward with int32 accumulator
10. `pkg/quantization/evaluator.go` — `SideBySideEvaluator[T]`

**Verify criterion (Phase β gate)**: `TestCalibrationRunner_TooFewSamples` returns `ErrCalibTooFewSamples`; `TestFullInt8Forward` verifies `||y_float − y_quant||_∞ ≤ 5%·||y_float||_∞` on a 64-neuron Dense with random weights; `TestSideBySideEvaluator` reports non-zero `PerLayerL2`.

### Phase γ — Attention Projection Quantization

11. `pkg/quantization/attn.go` — `QuantizedAttentionProjections[T]` (quantize Wq/Wk/Wv/Wo; scoring + softmax remain float)
12. `quantizer.go` (amend) — type-switch gains case for `*attention.MultiHeadAttention[T]`

**Verify criterion (Phase γ gate)**: `TestQuantizedAttentionForward` — FD numerical parity of quantized vs float projection forward on `Dmodel=64, NumHeads=4, SeqLen=8`; `||delta||_∞ ≤ 5%`.

## 6. Implementation Notes

1. **`roundHalfEven`** is not in the Go standard library — implement in `params.go` as: `math.Floor(x + 0.5)` with special case for `x + 0.5` being an integer: round to even. Test against L1 §4.2 and a round-at-0.5 table.
2. **Weight access from `*nn.Network[T]`**: the public API of `Network[T]` does not expose raw weight slices. `Quantize[T]` uses a type-switch on the `cfg.ConvPrefix` slice (accessible via `net.Config()`) and direct struct access per layer type — the same pattern used by `applyConvBackward` in `pkg/nn/train.go`. Acceptable coupling because quantization is a peer package in the same module.
3. **SHA-256 of `.nn.json`**: use `crypto/sha256` (stdlib). Hash the canonical JSON bytes of the source network, not the file path — ensures the hash is portable across filesystem moves.
4. **base64 for int8 arrays**: use `encoding/base64` (stdlib) `StdEncoding`. The byte array is the raw little-endian int8 buffer (Go `unsafe.Slice` or plain loop write into `[]byte`). No compression — `.qnn.json` size is dominated by int8 weights which are already at minimal bit depth.
5. **ConvPrefix access pattern**: `Quantize[T]` iterates `net.Config().ConvPrefix` in order; the index becomes the `QuantizedLayerJSON.Index` field. This preserves layer ordering in the `.qnn.json` array so `UnmarshalQNN` can reconstruct the inference chain.

## 7. Drawbacks & Alternatives

- **Alternative: Extend `layer.Layer[T]` with a `Quantize()` method** — rejected. Embedding quantization logic in every layer type couples the training infrastructure to deployment concerns. The peer-package transformer pattern (this spec) is cleaner and defers `pkg/layer/` changes indefinitely.
- **Alternative: Use `float32` (not `float64`) for scale / zero_point** — rejected. Calibration statistics accumulate across up to 1000 samples and across attention heads; float32 accumulation drift is measurable at this scale. The `scale []float64` field costs 8 bytes × Cout per layer — negligible vs the `[]int8` weight storage it accompanies.
- **Alternative: Eager int32 GEMM in Phase α** — not needed. Weight-only mode dequantizes to float before GEMM. This is slower than full int8 GEMM but correct and dependency-free. The acceleration benefit is in the backend kernel path (QUANT-9), which lands separately.
- **Alternative: Store scale as `[]T` (generic)** — rejected. `T` is `float32 | float64`; using `float64` for all scale storage prevents precision loss on the `float32` training variant while keeping one serialization path. The cast at inference time is a single `T(scale[c])` per channel.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[L1-QUANT]` | `.design/main/specifications/l1-quantization.md` | L1 contract — affine math, calibration protocol, persistence schema; MUST match exactly |
| `[PERSIST]` | `pkg/persistence/` | `.nn.json` baseline format; `BaselineHash` computed from canonical JSON bytes |
| `[NN-TRAIN]` | `pkg/nn/train.go` | `applyConvBackward` type-switch pattern reused by `Quantize[T]` to access ConvPrefix layers |
| `[NN-CONFIG]` | `pkg/nn/config.go` | `Config.ConvPrefix` slice — the ordered list of pre-sequence layers iterated by `Quantize[T]` |
| `[LAYER-DENSE]` | `pkg/layer/dense.go` | Dense weight layout (`Weights []T`, row-major `Cout × Cin`) — the quantization target in Phase α |
| `[LAYER-CONV]` | `pkg/layer/conv/` | Conv1D/Conv2D filter layout — quantization target in Phase α |
| `[LAYER-ATTN]` | `pkg/layer/attention/multihead.go` | MHA projection matrices `Wq/Wk/Wv/Wo` — quantization target in Phase γ |

<!-- Downstream agent instruction: Load [L1-QUANT] and [NN-TRAIN] BEFORE writing any code. The L1 §4.2 golden values (scale≈0.0236 for r_max=3.0, q(1.0)=42) are normative — the Phase α unit test MUST reproduce them exactly to validate roundHalfEven. The three-phase plan (α weight-only → β calibration + full-int8 → γ attention projections) is the authoritative delivery order; do not collapse phases. QUANT-C7 (no mixed precision in v0.1) means the pipeline mode is whole-network — no per-layer WeightOnly vs FullInt8 toggle in v0.1. -->

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-20 | Initial L2 spec authored via `/magic-spec` Spark 1. Implements all 10 L1 invariants (QUANT-1..QUANT-10) with full Invariant Compliance table. Package layout `pkg/quantization/` (10 files). Core types: `QuantizationConfig[T]`, `QuantizationParams`, `QuantizedNetwork[T]`, `QuantizedDense[T]`, `QuantizedConv1D/2D[T]`, `QuantizedAttentionProjections[T]`, `CalibrationRunner[T]`, `SideBySideEvaluator[T]`. Three-phase delivery plan (α weight-only Dense/Conv → β activation calibration + full-int8 GEMM → γ attention projection). GC-1..GC-5 Go-specific constraints: no external deps, `utils.Float` preserved, read-only network input, phase-gated scope, no `layer.Layer[T]` for quantized types. Promoted Draft → Stable via Trust Mode (C9): MVC satisfied (Overview + §3 Invariant Compliance + §4 Detailed Design); no RULES.md conflicts; no circular dependencies; layer constraint satisfied (Implements l1-quantization.md Stable v0.1.0); orthogonal to active Phase 18 transformer work. |
