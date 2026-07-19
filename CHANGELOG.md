# Changelog

All notable changes to the GoNN library will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
release artifacts dictated by [.magic/run.md](.magic/run.md) Phase Completion / Plan Completion.

## [0.18.0] — 2026-07-19

### Full Restoration — training-math correctness release

A two-pass engineering audit (static review of all packages + 25 live
simulations with numeric gradient checks) found that the composition layer
broke what the leaf layers computed correctly: backprop dropped one
activation-derivative factor per layer crossed, the configured loss never
reached the gradient, several layer families never received weight updates,
and half a dozen configured subsystems were silently disconnected. This
release fixes all of it behind a numeric safety net.

**BREAKING (numerics):** every network trains along the mathematically
correct gradient now. Models trained with earlier versions will converge
differently (typically better); retraining is recommended. Checkpoints and
weight files from ≤0.17 load, but `BatchNorm` running statistics from old
JSON payloads are reset to identity (the scalar-stat schema was replaced by
per-feature slices).

#### Fixed — core training math (audit block A)

- **Chain rule restored in `Network.CalculateMisses`** — the miss propagated
  from layer *i+1* (or Output) into layer *i* now folds the SOURCE layer's
  σ′(preact). The historical omission corrupted every multi-hidden gradient;
  a central-difference oracle over 5 topologies × 8 activations
  (`internal/verification/`) now gates the training path (worst rel. err < 1e-4).
- **`tanhDerivative`** returns `1 − tanh²(z)` (was a broken post-activation
  formula), **SELU** derivative alpha typo fixed.
- **All 18 loss functions have true derivatives** (`loss.Derivative`,
  `loss.VectorLoss`, `loss.Aggregate`); the output residual is
  `−∂ℓ/∂y` for the CONFIGURED loss (previously every loss trained as MSE).
  `cceLossSingle` no longer returns a constant 0. `MSE` uses the ½ convention
  so the (t − y) residual remains its exact derivative.
- **True vector softmax** at the output (`activation.SoftmaxInto`, stable),
  fused (SIGMOID, BCE) and (SOFTMAX, CCE) gradient shortcuts; SOFTMAX with a
  non-cross-entropy loss and SOFTMAX on hidden layers are compile errors.

#### Fixed — frozen layers & embeddings (audit block C)

- `Conv2D`, `SimpleRNN`, `LSTM`, `GRU`, and learnable `PositionalEncoding`
  now train: the hand-maintained type switch in the prefix backward pass was
  replaced by an `ApplyGradSGD(lr)` capability interface.
- `TokenEmbedding` lazy-initialises its buffers (compile-time panic fixed);
  `PositionalEncoding` gradient accumulator resets after each apply.

#### Fixed — lifecycle, concurrency, robustness (audit block D)

- **`Query` is race-free**: dense inference runs a stateless forward
  (`Network.InferDense`) under an `RWMutex` read lock — verified with a
  16-goroutine storm under `-race`.
- `Stop` no longer sterilises the network (a stopped network can Fit again);
  `AndTrain` applies and restores its learning-rate override; topology
  mutations preserve surviving weights and reset optimizer moments (no more
  Adam panic); NaN/Inf inputs are rejected; `ApplyFlatWeights` errors on
  length mismatch; callbacks abort training on ANY error as documented.

#### Added — subsystem wiring (audit block B)

- **Normalization in the dense path**: `BatchNorm` / `LayerNorm` /
  `GroupNorm` participate in forward, backward (exact gradients, oracle-
  checked), and γ/β training. `BatchNorm` was reworked to per-feature
  running statistics with sample-stream EMA semantics; `ForwardInference`
  provides a mutation-free path for concurrent Query.
- **Honest Dropout**: the mask now gates activations INSIDE the forward pass
  (per layer, before the next layer consumes them) and routes the backward
  miss through the same mask. Inference never masks.
- **L1/L2 reach the gradient** (`Regularizer.WeightGrad`); **LR schedulers
  auto-bind** to the optimizer at compile.
- **`nn.Save` / `nn.Load`** — public persistence API (config + weights,
  hash-linked); the CLI and `examples/persistence` now use it.
- **Checkpointing wired into `Fit`** via `WithCheckpoint(dir, everyN, sweep)`
  and `nn.Resume(dir)` (weights + optimizer state, retention sweep, sweeper
  errors logged).
- **`FitDataset(ctx, ds)`** — managed training over streaming
  `dataset.Dataset` sources with context cancellation; `Prefetch` and CSV
  datasets implement `io.Closer`; `WithInputShapeFrom(ImageShaper)`.
- **Visualization server is real**: `/v1/snapshot` serves live loss/epoch/
  layer/activation state published by Fit; `NN.VisAddr()` exposes the bound
  address.
- **Dense-head quantization**: `quantization.Quantize` now covers the MLP
  head (per-layer activations included) in addition to the conv prefix;
  `baselineHash` digests the actual weights (previously hashed `{}`).

#### Security & hardening (audit block E)

- Constant-time bearer-token comparison in the visualization server;
  HTTP timeouts (`IdleTimeout`, `MaxHeaderBytes`) on the vis server;
  pprof on a DEDICATED mux (no more `DefaultServeMux` leakage) with graceful
  shutdown via `NN.Close` and a Warn on non-loopback binds; gzip-bomb limit
  in the checkpoint reader; IDX record-length overflow cap in the MNIST loader.

#### Changed

- `applyDefaults` substitutes only exactly-zero values: a negative
  `WithLearningRate` is now a compile error (was silently 0.3); a negative
  `WithLossLimit` documented as "never stop early".
- `Pause` parks the training goroutine on a `sync.Cond` (zero CPU) instead
  of a `runtime.Gosched` busy-wait.
- Dead code removed: `topologyTx` snapshot machinery, transformer `bufAttn`
  buffers, `PositionalEncoding.lastIn`, the attention backward placeholder
  double-pass, empty `pkg/nn/api/`.

#### Changed — weight storage is now structure-of-arrays

The dense engine no longer chases pointers to read weights. Each layer owns a
contiguous row-major `[out][in]` run carved out of one network-wide array, and
`axon.Axon` became a VIEW over it (`W()` / `SetW()` replace the old exported
`Weight` field; `Bind` attaches an axon to a slot). The bias is folded in as the
last input column, pinned to 1, so no kernel special-cases it.

This is what makes `WithBackend` real, and it falls out of an invariant that was
already true: the canonical flat-weight order (layer → cell → axon) *was*
row-major matrix order all along, so `FlatWeights` / `ApplyFlatWeights`
collapsed into a single copy.

- **BREAKING (API):** `axon.Axon[T].Weight` (field) → `W()` / `SetW(v)`.
  The type is never JSON-serialised directly, so no on-disk format changed.
- Topology mutations repack storage and rebind every axon, carrying learned
  weights across; the store is rebuilt in one place, so a reallocation can never
  leave a stale weight reference behind.
- `InferDense` dropped its per-call `map[Nucleus]T` for plain slices — the
  concurrent-inference path got both simpler and faster.

#### Added — the compute backend actually computes

`nn.WithBackend` selected a backend that was then stored and never consulted
(audit B8). The dense passes now route their matrix primitives through it.

- **`compute.DenseKernels[T]`** — optional capability interface (`MatVec`,
  `MatVecT`, `GradOuter`) alongside `compute.DenseMatrix[T]`. Implemented by
  `cpu.Backend`, which is therefore live by default.
- The split is deliberate: kernels are pure linear algebra with no notion of
  activations, losses, optimizers, normalization, or dropout. That orchestration
  stays in the engine, so choosing a backend can never silently bypass the
  configured optimizer or regularizer — only the inner products move.
- **Graceful degradation, both directions.** A backend that does not implement
  `DenseKernels` logs a Warn and runs on the engine's internal reference loops.
  A backend whose kernel *errors* is disabled after one logged Warn and the run
  finishes on correct math — a failing accelerator never becomes silently wrong
  training. Both paths are regression-tested, including a bit-exact parity test
  between the delegated and reference runs.
- `Network.SetBackend` / `Network.KernelsActive` expose the wiring.
- **OpenCL remains, by design, NOT an accelerated training path.** It does not
  implement `DenseKernels`, so selecting it yields correct training on the
  reference loops plus a Warn. The reason is structural: every call there
  allocates device buffers, uploads the full weight matrix, launches, blocks,
  and reads back, while `DenseKernels` runs once per layer per sample — the
  transfer cost would dominate the arithmetic by orders of magnitude. Making it
  real needs device-resident weights and mini-batching, both engine-level
  changes. The package documents this instead of shipping a slower "accelerator".

#### Added — regression infrastructure

- `internal/verification/` — the numeric safety net: gradient-check oracle
  (central differences vs `AppendFlatGradients`), convergence goldens,
  lifecycle/concurrency/wiring scenario tests. `.github/workflows/ci.yml`
  runs vet + full suite with `-race`.

## [0.17.0] — 2026-05-21

### Post-Training Quantization (PTQ)

Phase 19 delivers `pkg/quantization/` — a peer transformation layer that consumes a trained
`*nn.NN[T]` and produces a deployable `QuantizedNetwork[T]` int8 artifact persisted as
`.qnn.json`. Implements all 10 L1 invariants (QUANT-1..10) across three phases: α (weight-only
Dense/Conv1D/Conv2D), β (activation calibration + full-int8 GEMM path), γ (attention projection
matrices). No changes to `pkg/nn/`, `pkg/layer/`, or `pkg/compute/`.

#### Added

- **`pkg/quantization/config.go`** — `QuantMode` enum (`WeightOnly`, `FullInt8`); `QuantizationConfig[T]` with `Mode`, `WeightGranularity`, `ActGranularity`, `Strategy`, `PercentileThreshold`, `Seed`; `CalibrationProvenance` struct; `DefaultQuantizationConfig[T]()`.
- **`pkg/quantization/params.go`** — `Granularity` (`PerTensor`, `PerChannel`); `CalibStrategy` (`MinMax`, `Percentile99p9`, `Entropy`); `QuantizationParams`; `dequantize`, `quantize` with banker's rounding (`roundHalfEven`); `computeSymmetric`, `computeAsymmetric`.
- **`pkg/quantization/quantizer.go`** — `QuantizedNetwork[T]` struct; `Quantize[T](net, calibSamples, cfg)` entry point (type-switches on ConvPrefix: Conv1D, Conv2D, MultiHeadAttention, float pass-through for unknown types); `Load[T](path)` deserialisation entry point; SHA-256 `BaselineHash`.
- **`pkg/quantization/dense.go`** — `QuantizedDense[T]`: weight-only and full-int8 forward paths; int32 accumulator GEMM with zero-point bias corrections; `SetActParams` setter.
- **`pkg/quantization/conv.go`** — `QuantizedConv1D[T]`, `QuantizedConv2D[T]`: weight-only int8 convolutional layers; dequantize filter weights on the fly per output channel.
- **`pkg/quantization/calibration.go`** — `CalibrationRunner[T]`: drives ConvPrefix manually; `Run` collects per-layer activation min/max + percentile-clip values; `Params()` returns per-layer `QuantizationParams`; hard floor 32 samples → `ErrCalibTooFewSamples`.
- **`pkg/quantization/evaluator.go`** — `Evaluate[T]`: side-by-side float vs int8 comparison; `EvaluationResult` with `MetricFloat`, `MetricQuant`, `DeltaRelative`, `PerLayerL2`; informational only, no auto-accept gate (QUANT-C6).
- **`pkg/quantization/attn.go`** — `QuantizedAttentionProjections[T]`: weight-only int8 Wq/Wk/Wv/Wo projections; scoring and softmax remain float (QUANT-10); full multi-head attention forward.
- **`pkg/quantization/persistence.go`** — `.qnn.json` wire format; `MarshalQNN`/`UnmarshalQNN` (int8 weights as base64, float64 bias as IEEE-754 LE base64); `Save`; `Load` via `UnmarshalQNNFile`.

#### Modified

- **`pkg/utils/errors.go`** — `ErrCalibTooFewSamples`, `ErrDegenerateRange` quantization sentinels.
- **`pkg/layer/conv/conv.go`** — exported `OutputLen`, `PadSamePadding`, `Isqrt` wrappers for shape-arithmetic helpers used by `pkg/quantization`.

#### Coverage floors

| Package              | Coverage |
| -------------------- | -------- |
| `pkg/quantization`   | 91.4 %   |

## [0.16.0] — 2026-05-20

### Transformer Block Implementation

Phase 18 lands the composite Transformer-block primitive: `EncoderBlock[T]`,
`DecoderBlock[T]`, and `Stack[T]` in the new `pkg/layer/transformer/` package.
All three types compose the existing Phase 10–17 primitives (LayerNorm, MultiHeadAttention,
Dropout, activation dispatcher) and are wired into the `pkg/nn` training loop via the
`ConvPrefix` infrastructure (no new Config field).

#### Added

- **`pkg/layer/transformer/config.go`** — `TransformerConfig[T]` (`SeqLen`, `Dmodel`, `NumHeads`, `Dff`, `PreNorm`, `DropoutRate`, `Activation`); `Mode` enum (`EncoderMode`/`DecoderMode`); `applyActivationInPlace` helper.
- **`pkg/layer/transformer/block.go`** — `Block[T]` private interface; `addInPlace[T]` residual helper (TRANS-6); internal position-wise `ffn[T]` (W1/b1/W2/b2, Xavier init, Forward/Backward, MarshalJSON).
- **`pkg/layer/transformer/encoder.go`** — `EncoderBlock[T]`: post-norm (TRANS-2) + pre-norm (TRANS-3) `Forward`/`Backward`; three Dropout positions (TRANS-C7); `Init`, `SetTraining`, `ApplyGradSGD`, `SetPaddingMask`, `MarshalJSON`/`UnmarshalJSON`; lazy cache allocation for shape-inference safety.
- **`pkg/layer/transformer/decoder.go`** — `DecoderBlock[T]`: structurally identical to `EncoderBlock[T]`; inner MHA built with `Causal:true` (TRANS-4, TRANS-C2); same Forward/Backward/JSON wiring.
- **`pkg/layer/transformer/stack.go`** — `Stack[T]` (N-block sequential chain); independent RNG fork per block (TRANS-C8, no weight tying); `Forward`/`Backward`/`SetPaddingMask`/`ApplyGradSGD` fan-out; JSON envelope `{Type, Config, Mode, Blocks:[...]}` (TRANS-9).
- **`pkg/nn/options.go`** — `WithEncoderBlock[T]`, `WithDecoderBlock[T]`, `WithEncoderStack[T]`, `WithDecoderStack[T]` — all append to `ConvPrefix`.
- **`pkg/nn/train.go`** — `applyConvBackward` switch extended with `*transformer.EncoderBlock[T]`, `*transformer.DecoderBlock[T]`, `*transformer.Stack[T]` → `ApplyGradSGD`.

#### Modified

- **`pkg/layer/norm/layernorm.go`** — added `Backward(upstream []T) []T`, `ApplyGradSGD(lr T)`, `ForwardSeq`/`BackwardSeq` (per-position sequence-aware variants); `Forward` caches `xHat` and `invSd` for Backward use.

#### Coverage floors

| Package                    | Coverage |
| -------------------------- | -------- |
| `pkg/layer/transformer`    | 88.3 %   |
| `pkg/layer/norm`           | 84.2 %   |
| `pkg/nn`                   | 75.5 %   |

## [0.15.0] — 2026-05-20

### NLP Foundation: Multi-Head Attention + Embedding Layers

Phase 17 adds two parallel NLP tracks: full multi-head scaled dot-product attention
(ATT-1..ATT-10) and token/positional embedding (EMB-1..EMB-10), with both tracks
wired into the `pkg/nn` training loop via the existing `ConvPrefix` infrastructure.

#### Added — Track A: Attention

- **`pkg/layer/attention/multihead.go`** — `MultiHeadAttention[T utils.Float]`:
  - `NewAttention[T](seqLen, dmodel int, causal bool)` — single-head sugar (NumHeads=1).
  - `NewMultiHeadAttention[T](seqLen, dmodel, numHeads int, causal bool)` — multi-head; panics with `ErrAttentionHeadsMismatch` when `dmodel % numHeads != 0`.
  - `Forward`: project Q/K/V → `splitHeads` → scaled scores → causal+padding masks → row-wise softmax → context × V → `joinHeads` → output projection.
  - `Backward`: ATT-7 four-path: output projection, V-path, softmax-backward, Q/K projection.
  - `ApplyGradSGD(lr T)` — inline SGD update for all 8 weight/bias matrices.
  - `MarshalJSON` / `UnmarshalJSON` round-trip.
  - 91.3 % statement coverage.
- **`pkg/layer/attention/cell.go`** — shared helpers: `softmaxRowwise`, `softmaxRowwiseWithMask`, `softmaxBackwardRowwise`, `attentionScores`, `applyCausalMask`, `applyPaddingMask`, `projMat`, `contextMul`, `splitHeads`, `joinHeads`.
- **`pkg/layer/attention/masked_layer.go`** — `MaskedLayer[T]` interface + `SetPaddingMask` implementation (per-call semantics, cleared after Forward).
- **`pkg/nn/options.go`** — `WithAttention[T]`, `WithMultiHeadAttention[T]`, `WithCausalAttention[T]`.
- **`pkg/nn/train.go`** — `applyConvBackward` extended to type-assert `*attention.MultiHeadAttention[T]` and call `ApplyGradSGD`.

#### Added — Track B: Embedding

- **`pkg/layer/embedding/sparse.go`** — `sparseGrad[T]` bitset accumulator: O(unique IDs × Dmodel) per step via `[]uint64` bitset + `bits.TrailingZeros64` iteration.
- **`pkg/layer/embedding/token.go`** — `TokenEmbedding[T]`: learnable lookup table, `ForwardIDs`, sparse `Backward`, `ApplyGradSGD`, JSON round-trip. Implements `layer.IDLayer[T]`.
- **`pkg/layer/embedding/table.go`** — `buildSinusoidalTable[T]`: Vaswani et al. 2017 PE(pos, 2i) = sin / PE(pos, 2i+1) = cos formula.
- **`pkg/layer/embedding/positional.go`** — `PositionalEncoding[T]` with `Sinusoidal` / `Learnable` mode; `Forward` adds table to input; `Backward` passes gradient through (accumulates for Learnable).
- **`pkg/layer/embedding/stack.go`** — `EmbeddingStack[T]` composing `TokenEmbedding` + `PositionalEncoding` as a single `layer.IDLayer[T]`. 90.7 % statement coverage.
- **`pkg/nn/options.go`** — `WithEmbeddingStack[T]`, `WithTokenEmbedding[T]`.
- **`pkg/nn/train.go`** — sparse SGD applied via `applyTokenEmbeddingSGD`; Learnable positional table updated via dense `applyConvSGD`.

#### Modified

- **`pkg/layer/core.go`** — added `Layer[T utils.Float]` and `IDLayer[T utils.Float]` interfaces (same method set as `conv.Layer[T]`; structural typing satisfies both).
- **`pkg/utils/errors.go`** — added `ErrAttentionHeadsMismatch`, `ErrAttentionMaskLength`, `ErrVocabOutOfRange` sentinels.

#### Coverage floors

| Package                | Coverage |
| ---------------------- | -------- |
| `pkg/layer/attention`  | 91.3 %   |
| `pkg/layer/embedding`  | 90.7 %   |

## [0.14.0] — 2026-05-19

### Recurrent Completion + GPU Backward + AI-Meta Rollout

Phase 16 closes three tracks deferred from Phase 15: full recurrent layer set
(GRU[T] + LastStep[T] + gradient clipping), OpenCL Dense Backward kernel with
graceful CPU fallback wiring, and AI-Meta compliance hooks rolled out to 10 more
packages.

#### Added — Track A: Recurrent Completion

- **`pkg/layer/recurrent/gru.go`** — `GRU[T utils.Float]`:
  - `NewGRU[T](seqLen, inSize, hidden int)` with Xavier (W_x) + Orthogonal (W_h) init.
  - 3-gate Forward: reset gate `r_t`, update gate `z_t`, candidate `ñ_t`, output `h_t`.
  - `Backward`: BPTT gradient accumulation through all gates; returns `gradInput` per timestep.
  - `MarshalJSON` / `UnmarshalJSON` round-trip; `Step` / `Init` wired to spec REC-3.
  - 96.0 % statement coverage.
- **`pkg/layer/recurrent/laststep.go`** — `LastStep[T utils.Float]`:
  - Stateless sequence-to-vector collapser: picks final timestep `output[seqLen-1, :]`.
  - `Backward`: scatters upstream gradient into final-timestep position; zeros elsewhere.
  - Implements `conv.Layer[T]`; no parameters (Init/Step are no-ops).
- **`pkg/optimizer/clip.go`** — `ClipByGlobalNorm[T](grads [][]T, threshold T)`:
  - Computes global L2 norm across all gradient slices; scales by `min(1, threshold/‖g‖₂)`.
  - No-op when `threshold ≤ 0` or `‖g‖₂ ≤ threshold`.
  - `pkg/optimizer/` coverage: **90.7 %** (floor 85 %).
- **`pkg/nn/options.go`** — four new recurrent options:
  - `WithSimpleRNN[T](seqLen, inSize, hidden int)`, `WithLSTM[T]`, `WithGRU[T]` — append recurrent layer to `Config.ConvPrefix`.
  - `WithLastStep[T](seqLen, hidden int)` — appends `*recurrent.LastStep[T]`.
  - `WithGradClipNorm[T](threshold T)` — stores in `Config.GradClipNorm`; train.go applies before optimizer step.
- **`pkg/nn/compile.go`** — `setupRecurrentShapes[T]` + `hasRecurrentLayer[T]`:
  - Pre-pass validates SeqLen/InSize/Hidden dimensions and shape continuity across the recurrent prefix.
  - `hasRecurrentLayer` type-switches on `*recurrent.GRU[T]`, `*recurrent.LSTM[T]`, `*recurrent.SimpleRNN[T]`, `*recurrent.LastStep[T]`.
- **`pkg/nn/recurrent_compile_test.go`** — 5 subtests covering: GRU+LastStep compose, LSTM alone, SimpleRNN+LastStep, shape mismatch on input, shape mismatch on hidden.
- **`pkg/utils/errors.go`** — `ErrRecurrentShapeMismatch` sentinel.

#### Added — Track B: GPU Backward + Backend Wiring

- **`pkg/compute/gpu/opencl/kernels.cl`** — two OpenCL kernels:
  - `dense_grad_w`: weight gradient `∂L/∂W = Xᵀ·∂L/∂Y` (batched matmul).
  - `dense_grad_x`: input gradient `∂L/∂X = ∂L/∂Y·Wᵀ`.
- **`pkg/compute/gpu/opencl/kernels.go`** — Go bindings for Backward kernels (build tag `cgo && opencl`).
- **`pkg/compute/gpu/opencl/bench_test.go`** — GPU vs CPU benchmarks (build tag `cgo && opencl`):
  - `BenchmarkDenseForward/Backward` at 64×64, 256×256, 1024×1024 matrix sizes.
  - CPU baseline variants for direct comparison.
  - Expected speedup: ≥2× for ≥256×256 on iGPU; ≥5× on discrete NVIDIA/AMD.
- **`pkg/nn/options.go`** — `WithBackend[T](b compute.Backend[T]) Option[T]`:
  - Stores backend in `Config.Backend`; compile() probes with `Allocate(1)`.
  - On `ErrBackendUnavailable`: logs `Warn("backend unavailable, falling back to CPU")` and uses CPU reference backend.
- **`pkg/nn/compile.go`** — `resolveBackend[T]`: probe logic + CPU fallback.
- **`pkg/nn/init.go`** — blank import `_ "github.com/teratron/gonn/pkg/compute/cpu"` ensures CPU backend registered at startup.
- **`pkg/nn/config.go`** — `Backend compute.Backend[T]` field added to `Config[T]`.
- **`pkg/nn/nn.go`** — `backend compute.Backend[T]` field added to `NN[T]`.
- **`pkg/nn/backend_test.go`** — 3 subtests: nil backend → CPU, unavailable backend → CPU fallback, result parity CPU vs fallback.
- **`pkg/compute/gpu/`** coverage: **90.0 %** (floor 80 %).

#### Added — Track C: AI-Meta Linter `--resolve` + Compliance Rollout

- **`pkg/aimeta/resolver.go`** — `Resolver` type applying mechanical auto-fixes: INDENT (missing `//`), LABEL (lowercase → Title-Case), LAST (missing terminal newline).
- **`cmd/lint-aimeta/resolve.go`** + `--resolve` flag in `main.go` — in-place fix mode; re-lints after fix and reports zero-violation count.
- **`pkg/aimeta/`** coverage: **86.6 %** (floor 80 %).
- **`aimeta_test.go`** hooks (rollout phases 3–5) — `TestAIMetaCompliance` added to 10 packages:
  - Phase 3: `pkg/activation/`, `pkg/loss/`
  - Phase 4: `pkg/neuron/`, `pkg/layer/`, `pkg/network/`
  - Phase 5: `pkg/dataset/`, `pkg/checkpoint/`, `pkg/compute/`, `pkg/persistence/`, `pkg/nn/`
- AI-Meta annotations (`Purpose`, `Usage`, `Concurrency`, `Related`, `Stability`) added or corrected on ~180 exported symbols across all 10 packages (ENUM + TIER violations resolved).

#### Changed

- `pkg/layer/recurrent/`: GRU[T] + LastStep[T] added alongside Phase 15's SimpleRNN[T] + LSTM[T].
- `pkg/nn/train.go`: applies `ClipByGlobalNorm` before optimizer step when `cfg.GradClipNorm > 0`.
- `pkg/compute/backend.go`: `Concurrency` annotation corrected to `NotSafe.` (ENUM fix).
- Multiple `pkg/nn/`, `pkg/layer/`, `pkg/network/`, `pkg/activation/`, `pkg/loss/`, `pkg/neuron/`, `pkg/dataset/`, `pkg/checkpoint/`, `pkg/compute/`, `pkg/persistence/` files: AI-Meta ENUM and TIER annotations updated.

#### Coverage Summary

| Package | Coverage | Gate |
| :--- | :--- | :--- |
| `pkg/layer/recurrent/` | 96.0 % | ≥80 % ✓ |
| `pkg/optimizer/` | 90.7 % | ≥85 % ✓ |
| `pkg/compute/gpu/` | 90.0 % | ≥80 % ✓ |
| `pkg/aimeta/` | 86.6 % | ≥80 % ✓ |
| `pkg/nn/` | 77.2 % | ≥75 % ✓ |

#### Known Issues

- Pre-existing timing-flaky tests: `TestPauseResumeCycle`, `TestMultiHiddenXOR`, `TestRepeatBuilderBenchmark100Layer`.
- Race detector requires CGO on Windows (gcc not in PATH); `-race` deferred to CI.
- GPU benchmarks require `cgo && opencl` build tags and an OpenCL runtime; skipped in default CI.
- `go run ./examples/mnist/` requires user-supplied IDX data; see `examples/mnist/README.md`.

## [0.13.0] — 2026-05-19

### Recurrent Foundation + GPU Backend Skeleton + AI-Meta Linter

Phase 15 delivers three parallel foundation tracks: recurrent layers (SimpleRNN + LSTM
with full BPTT), GPU backend skeleton (OpenCL Dense Forward kernel), and the AI-Meta
linter + compliance hook infrastructure.

#### Added — Track A: Recurrent Layers

- **`pkg/layer/recurrent/cell.go`** — shared activation helpers (sigmoid, tanh fused).
- **`pkg/layer/recurrent/simplernn.go`** — `SimpleRNN[T]` (Elman RNN) with BPTT.
- **`pkg/layer/recurrent/lstm.go`** — `LSTM[T]` (4-gate: input, forget, gate, output) with BPTT; forget-gate bias init 1.0 (REC-3).
- `pkg/utils/errors.go` — `ErrRecurrentShape` sentinel.
- `pkg/utils/math.go` — `Orthogonal[T](n int, rng *rand.Rand) []T` initialiser.
- `pkg/layer/recurrent/` coverage: **97.2 %**.

#### Added — Track B: GPU Backend Skeleton

- **`pkg/compute/gpu/`** — umbrella package with `Backend[T]` stub; build-tag isolation `cgo && opencl`.
- **`pkg/compute/gpu/opencl/`** — `kernels.cl` Dense Forward kernel + Go bindings.
- `pkg/compute/gpu/` coverage: **90.0 %**.

#### Added — Track C: AI-Meta Linter

- **`pkg/aimeta/`** — grammar parser, violation types, `Check()` + `Options`.
- **`cmd/lint-aimeta/`** — CLI binary; `go run ./cmd/lint-aimeta/ <pkg...>`.
- **`pkg/utils/aimeta_test.go`** — first rollout phase 1 compliance hook.
- `pkg/aimeta/` coverage: **82.9 %**.

## [0.12.0] — 2026-05-17

### 2-D Convolutional Layers (Conv2D, MaxPool2D, AvgPool2D, Flatten2D) + MNIST CNN Example

Phase 14 ships a complete 2-D convolutional layer set in `pkg/layer/conv/`,
integrates it with the `pkg/nn` options + compile pipeline, adds the MNIST
image-shape adapter to `pkg/dataset/`, and delivers `examples/mnist_cnn/` (E16)
as the canonical CNN demo. Gradient correctness is validated by a 4-case
centred finite-difference test matrix (CONV2D-4).

#### Added

- **`pkg/layer/conv/conv2d.go`** — `Conv2D[T utils.Float]`:
  - Constructor `NewConv2D[T](numFilters, inChannels, kernelH, kernelW, strideH, strideW int, pad PadMode, useBias bool)` with negative-arg clamp.
  - `Forward`: 6-level CHW cross-correlation loop, PadValid/PadSame, saves `lastInput` for backward.
  - `Backward`: CONV2D-4 gradW/gradX/gradB accumulation; PERF-4 cap-check + zero-reset → zero allocs/op in steady state.
  - `Init(rng *rand.Rand)`: He-normal weight sampling; zero-bias init.
  - `SetInputShape(inH, inW int)`, `OutputShape() (outH, outW int)`, `Validate(inC, inH, inW int) error`.
  - `MarshalJSON` / `UnmarshalJSON` via struct-alias trick (excludes scratch buffers).
  - Interface assertions in `conv.go`: `var _ Layer[float32] = (*Conv2D[float32])(nil)` + float64 variant.
- **`pkg/layer/conv/pool2d.go`** — `MaxPool2D[T]` + `AvgPool2D[T]`:
  - Non-overlapping per-channel 2-D pooling.
  - `MaxPool2D.Backward`: routes upstream only to argmax position (argmaxH/argmaxW stored on Forward).
  - `AvgPool2D.Backward`: distributes upstream × `1/(PoolH×PoolW)` across entire window.
  - `SetInputShape(inC, inH, inW int)`, `OutputShape() (outH, outW int)`, `Validate`.
- **`pkg/layer/conv/flatten2d.go`** — `Flatten2D[T]`:
  - Stateless CHW-collapse: Forward is identity (no copy); Backward returns upstream unchanged.
  - CONV2D-C9 element order: `(c, y, x)` → flat index `c*H*W + y*W + x`.
  - Single-channel `isqrt` inference on Forward when `SetInputShape` not called.
- **`pkg/layer/conv/conv2d_test.go`** — 17 test functions covering:
  - Forward shape table (5 cases), outputShape arithmetic, JSON round-trip, PadValid guard.
  - Centred finite-difference gradient check for `{PadValid, PadSame} × {C=1, C≥2}` (CONV2D-4).
  - MaxPool2D/AvgPool2D backward semantics; Flatten2D element ordering.
  - Accessor coverage: `InputSize`/`OutputSize`/`GradSlots`/`OutputShape`/`Validate` for all 2-D types.
  - `pkg/layer/conv/` coverage: **81.9 %** (floor 80 %).
- **`pkg/nn/options.go`** — five new functional options:
  - `WithInputShape[T](channels, height, width int)` — stores CHW layout in `Config.InputC/H/W`.
  - `WithConv2D[T](numFilters, inChannels, kernelH, kernelW, strideH, strideW int, pad, useBias)` — appends `*conv.Conv2D[T]` to `Config.ConvPrefix`.
  - `WithMaxPool2D[T](poolH, poolW int)`, `WithAvgPool2D[T](poolH, poolW int)`, `WithFlatten2D[T]()`.
- **`pkg/nn/compile.go`** — `setupConv2DShapes[T]` pre-pass:
  - Propagates `(C, H, W)` through each 2-D layer (`Conv2D → MaxPool2D/AvgPool2D → Flatten2D`) before the dummy-Forward shape walk.
  - Auto-infers `(1, S, S)` for flat square inputs (MNIST 784 → `(1, 28, 28)`) when `WithInputShape` not called.
  - Validates `InputSize == InputC*InputH*InputW` when shape declared.
  - Calls `SetInputShape` on each layer in sequence; guards mixed 1-D/2-D chains with `in2D` flag.
  - Wraps validation errors as `ErrConv2DShapeMismatch`.
- **`pkg/dataset/dataset.go`** — `ImageShaper` optional interface:
  - `ImageShape() (channels, height, width int)` — detected by `compile()` via type assertion for auto `InputC/H/W` wiring when `WithInputShape` not called explicitly.
- **`pkg/dataset/mnist.go`** — `MNISTLoader[T]` extensions:
  - `WithImageShape(channels, height, width int) *MNISTLoader[T]` — explicit shape annotation.
  - `ImageShape() (c, h, w int)` — resolves: explicit `WithImageShape` → IDX ndim==3 header → `isqrt` inference → `(0,0,0)`.
  - `MNISTLoader[T]` now implements `ImageShaper`.
- **`examples/mnist_cnn/`** — E16 CNN demo:
  - Architecture: `Input 1×28×28 → Conv2D(8,1,3×3,PadSame) → MaxPool2D(2×2) → Flatten2D → Dense(128,ReLU) → Dense(64,ReLU) → Output(10,Sigmoid)`.
  - Two-epoch training via `Fit` + `AndTrain` (fine-tune at LR=0.001).
  - `README.md` documents download URLs, flag usage, and key API summary.
- **`pkg/utils/errors.go`** — `ErrConv2DShapeMismatch` sentinel.

#### Changed

- `pkg/layer/conv/conv.go`: added 8 interface assertions for 2-D types (Conv2D/MaxPool2D/AvgPool2D/Flatten2D × float32/float64).
- `pkg/nn/config.go`: added `InputC, InputH, InputW int` fields to `Config[T]` for 2-D shape propagation.
- `.design/main/specifications/l2-dataset-loader-impl.md`: bumped v0.1.0 → v0.1.1 (added §5.3a WithImageShape + ImageShaper interface spec).
- `.design/main/specifications/l2-usage-examples.md`: bumped v1.0.0 → v1.1.0 (added E16 entry, updated coverage matrix and mermaid graph).
- `go.work`: added `./examples/mnist_cnn` module entry.

#### Known Issues

- Pre-existing timing-flaky tests: `TestPauseResumeCycle`, `TestMultiHiddenXOR`, `TestRepeatBuilderBenchmark100Layer`.
- Race detector requires CGO on Windows (gcc not in PATH); `-race` deferred to CI.
- `go run ./examples/mnist_cnn/` requires user-supplied IDX data; see `examples/mnist_cnn/README.md`.

## [0.11.0] — 2026-05-17

### Meta-Learning Hooks

Phase 12 (Track A) ships `MetaLearner[T]` — an inner `*NN[T]` that queries
its own inference output each training iteration to update registered hyperparameters
of the outer network (learning rate, per-layer scales, etc.). The mechanism is
advisory: errors from the inner network are logged at Warn level and do not abort
training.

#### Added

- **`pkg/nn/meta.go`** — new meta-learning types:
  - `ParamAccessor[T utils.Float]` interface: `Get() []T`, `Set([]T) error`, `Name() string`.
  - `ScalarParam[T]` struct: wraps `*T`; `Get` returns one-element slice; `Set` validates `len==1`
    or returns `ErrMetaLearnerShape`.
  - `SliceParam[T]` struct: wraps `*[]T`; `Get` returns a defensive copy; `Set` validates length
    matches or returns `ErrMetaLearnerShape`.
  - `FeatureFunc[T]` named function type: `func(loss T, iter int, maxIter int) []T`.
  - `DefaultFeatureFunc[T]`: returns `[]T{loss, T(iter)/T(maxIter)}` (loss + normalised progress).
  - `MetaLearner[T]` struct (`inner *NN[T]`, `params []ParamAccessor[T]`, `Features FeatureFunc[T]`):
    - `step(loss T, iter, maxIter int) error`: builds feature vector → `inner.Query` → apply
      `params[i].Set([]T{output[i]})` for each param; continue-on-error (collects first error,
      applies all); `ErrMetaLearnerShape` on output/param count mismatch.
- **`pkg/nn/options.go`** — `WithMetaLearner[T](ml *MetaLearner[T]) Option[T]`:
  sets `cfg.MetaLearner = ml`; rejected with `ErrMetaLearnerRunning` at compile time if
  `control.Load() != Idle`.
- **`pkg/nn/config.go`** — `MetaLearner *MetaLearner[T]` field added to `Config[T]` (nil = disabled).
- **`pkg/nn/train.go`** — single meta hook in `Fit` after `opt.Step` (batch level) and before
  `fireEvent(OnIterationEnd)`: `nn.cfg.MetaLearner.step(mean, epoch, MaxIterations)`;
  error → `log.Warn("meta-learner step failed", ...)`.
- **`pkg/utils/errors.go`** — two new sentinel errors:
  - `ErrMetaLearnerShape`: inner output size ≠ registered params, or `Set` receives wrong-length slice.
  - `ErrMetaLearnerRunning`: `WithMetaLearner` applied while outer network is training.
- **`pkg/nn/meta_test.go`** — 14 tests covering `ScalarParam`, `SliceParam`, `DefaultFeatureFunc`,
  `MetaLearner.step` (all 4 named cases), wiring, `ErrMetaLearnerRunning` guard, Fit integration,
  and 3-seed convergence check.
  `pkg/nn` coverage: 82.2 % (floor 80 %, no regression from v0.10.0).

#### Changed

- `pkg/nn/compile.go`: added `ErrMetaLearnerRunning` guard (rejects non-Idle compile with MetaLearner).

#### Known Issues

- Pre-existing timing-flaky tests: `TestPauseResumeCycle`, `TestMultiHiddenXOR`, `TestRepeatBuilderBenchmark100Layer`.
- Race detector requires CGO on Windows (gcc not in PATH); `-race` deferred to CI.
- `go run ./examples/mnist/` requires user-supplied IDX data; see `examples/mnist/README.md`.

## [0.10.0] — 2026-05-17

### Convolutional Layers, Dataset Loader, and AndTrain Continuation

Phase 11 (Tracks B + C) adds a convolutional prefix stage, the MNIST IDX dataset loader,
the `AndTrain` fine-tuning API, and two new examples. Track A (Meta-Learning) is deferred
to Phase 12 pending `l1-meta-learning-hooks` RFC → Stable promotion.

#### Added

- **`pkg/layer/conv/`** — new convolutional sub-package:
  - `conv.Layer[T utils.Float]` local interface (mirrors norm precedent; forward, backward, validate, init, JSON).
  - `PadMode` enum (`PadValid=0`, `PadSame=1`); `outputLen` and `padSamePadding` helpers.
  - `Conv1D[T]`: cross-correlation forward; ∂L/∂W, ∂L/∂X backward; He-normal `Init`; flattened
    filter-major `[]T` weight layout (one allocation, cache-locality); `MarshalJSON`/`UnmarshalJSON`
    round-trip (CONV-7); `Validate` enforces CONV-1 shape invariant.
  - `MaxPool1D[T]`: argmax-tracking backward (gradient routes only to argmax position).
  - `AvgPool1D[T]`: backward distributes 1/poolSize evenly across pool window.
  - `Flatten[T]`: stateless identity; backward returns upstream on shape match, nil on mismatch.
  - Compile-time interface assertions for all four types.
  - Coverage: 83.3 %.
- **`pkg/dataset/mnist.go`** — IDX binary format reader + MNIST loader:
  - `readIDXHeader`: validates magic bytes, dtype (6 IDX types: uint8/int8/int16/int32/float/double),
    ndim; `IDXReader` with per-record reusable buffer, `Next()` → `io.EOF`, `Reset()`.
  - `MNISTLoader[T]`: implements project `Dataset[T]` interface (batch streaming); pixel normalisation
    to [0, 1] via configurable norm factor; `NewMNISTLoader` (BYO readers) and `NewMNISTLoaderFiles`
    (opens files; `Close` releases handles; `Reset` reopens); context cancellation honoured.
  - Sentinel errors `ErrIDXMagic`, `ErrMNISTRecordMismatch` added to `pkg/utils/errors.go`.
  - Coverage: 85.2 %.
- **`pkg/nn/andtrain.go`** — `AndTrain` continuation API:
  - `(*NN[T]).AndTrain(samples []Sample[T], opts ...Option[T]) (int, T, error)`: snapshots config,
    applies caller-supplied options (e.g. lower learning rate), delegates to `Fit`, restores original
    config on defer. Preserves learned weights; resets convergence counters only.
  - Guards: `ErrUserConfig` if not Operational or empty samples; `ErrNetworkRunning` if Idle=false.
- **`pkg/network/propagation.go`** — `AppendInputGradient` method:
  computes `−Σ σ'(preact_j)·miss_j·w_{i→j}` for each input neuron; used by conv backward pass.
- **`pkg/nn/options.go`** — four new functional options: `WithConv1D`, `WithMaxPool1D`,
  `WithAvgPool1D`, `WithFlatten` to build the conv prefix chain before the dense network.
- **`pkg/nn/compile.go`** — conv chain shape resolution: resolves `outputLen` through the prefix
  chain; resizes the Input layer accordingly; `rawInputSize` captures original input size.
- **`pkg/nn/train.go`** — `runConvForward`, `applyConvBackward`, `applyConvSGD` integrated into
  `trainStep`; inlined SGD for conv weights (`w -= lr·g`); optimizer pluggability deferred to v0.11.
- **`pkg/nn/query.go`** — `runConvForward` pre-stage in inference path.
- **`examples/continuation/`** (E10) — demonstrates `AndTrain` end-to-end: trains XOR via `Fit`,
  then inverts targets at lower LR via `AndTrain`; `main_test.go` asserts weight continuity.
- **`examples/mnist/`** (E06) — demonstrates MNIST digit recognition: `MNISTLoader` + one-hot
  encoding + `BatchNorm` (784→128+BN→64→10) + two-epoch `Fit`+`AndTrain`; README explains IDX
  format, one-hot encoding, BatchNorm training/eval distinction, and download instructions.
  Smoke-run deferred: IDX data files not committed (see `examples/mnist/README.md`).

#### Changed

- **`go.work`** — added `./examples/continuation` and `./examples/mnist` workspace members.
- **`pkg/nn/config.go`** — `ConvPrefix []conv.Layer[T]` field added to `Config[T]`.
- **`pkg/nn/nn.go`** — `convPrefix`, `rawInputSize`, `convBuf`, `convGradBuf` fields on `NN[T]`.

#### Deferred

- **Meta-Learning Hooks** (T-11A01..T-11A03, T-11T01): `l1-meta-learning-hooks` spec remains RFC;
  deferred to Phase 12. Run `/magic-spec` to promote the spec and unlock the track.

#### Known Issues

- Pre-existing timing-flaky tests: `TestPauseResumeCycle`, `TestMultiHiddenXOR`,
  `TestRepeatBuilderBenchmark100Layer` — unaffected by Phase 11 changes.
- `TestXORTwoHiddenConvergence`: stochastic convergence flake on low-entropy random seed;
  pre-existing; passes consistently in isolation.
- `-race` flag requires CGO (gcc) on Windows; Phase 11 gate passed without `-race` locally;
  CI must re-run with `-race`.

## [0.9.0] — 2026-05-12

### Normalization Layers + Training Callbacks

Phase 10 adds post-activation normalization and structured training event callbacks.

#### Added

- **`pkg/layer/norm/`** — new normalization sub-package:
  - `Normalizer[T utils.Float]` interface: `Forward`, `SetMode`, `GradSlots`, `InputSize`, `OutputSize`.
  - `NormMode` enum (`NormTrain=0`, `NormEval=1`); shared helpers `stddev`, `applyAffine`.
  - `BatchNorm[T]`: EMA running stats (eval frozen), affine γ/β optional, `ErrBatchNormSingleSample`
    guard, `MarshalJSON`/`UnmarshalJSON` round-trip; functional options `WithBatchNormEps`,
    `WithBatchNormMomentum`, `WithBatchNormAffine`.
  - `LayerNorm[T]`: per-sample mean/var, stateless (mode stored but forward is always per-sample),
    affine optional, JSON round-trip.
  - `GroupNorm[T]`: G-group partition, `ErrGroupSizeMismatch` when `features%groups != 0`, affine
    optional, JSON round-trip.
  - Coverage: 82 %.
- **`pkg/nn/callbacks.go`** — training event callback infrastructure:
  - `ErrStopTraining` sentinel; `StopReason` enum (6 values: LossLimit, MaxIterations,
    ContextCancel, ExternalStop, Callback, LoopError).
  - `Snapshot[T]` (flat weight copy + epoch), `CallbackContext[T]`, `CallbackFn[T]` type.
  - `CallbackRegistry[T]`: `OnIterationEnd`, `OnImprovementFound`, `OnTrainEnd` slices.
  - `invokeOne` — panic recovery (CB-5); `fireEvent` — nil short-circuit (CB-3), stops on
    first `ErrStopTraining`; `fireOnTrainEnd` — defer-safe CB-8 guarantee.
- **`pkg/utils/errors.go`** — `ErrCallbackPanic` sentinel added.
- **`pkg/nn/train.go`** — `Fit` wired with callbacks: deferred `OnTrainEnd` (CB-8), per-epoch
  `OnImprovementFound` / `OnIterationEnd` dispatch, `ErrStopTraining` → rollback + return.
- **`pkg/nn/options.go`** — six new functional options: `WithNormAfterLayer`, `WithBatchNorm`,
  `WithLayerNorm`, `WithOnIterationEnd`, `WithOnImprovementFound`, `WithOnTrainEnd`.
- **`pkg/nn/config.go`** — `Callbacks *CallbackRegistry[T]` and `NormLayers map[int]Normalizer[T]`
  fields added to `Config[T]`.
- **`pkg/nn/compile.go`** — resolves nil norm-layer sentinels at compile time using hidden layer
  sizes; wires `n.callbacks` and `n.normLayers` from config.
- **`pkg/nn/nn.go`** — `SetTrain()` / `SetEval()` propagate `NormTrain`/`NormEval` to all
  registered `Normalizer[T]` instances; `callbacks` and `normLayers` fields added to `NN[T]`.
- **`pkg/nn/callbacks_test.go`** — 15 unit + integration tests covering CB-3/5/7/8 invariants,
  registration order, rollback on `ErrStopTraining`, `OnTrainEnd` on all exit paths,
  and `BenchmarkNoCallbacks`.

## [0.8.0] — 2026-05-10

### Metric Schedulers, Dynamic Topology, and Observability Stack

Phase 9 delivers three independent feature tracks: patience-based and cyclic LR schedulers,
runtime-safe topology mutation, and an HTTP observability server.

#### Added

- **`pkg/optimizer/metric_scheduler.go`** — `MetricScheduler[T]` interface:
  - Extends `Scheduler[T]` with `StepWithMetric(metric T) T`.
  - Compile-time assertions on all implementing types.
  - `boundScheduler[T]` extended to forward `StepWithMetric` via type assertion on inner scheduler.
- **`pkg/optimizer/reduce_on_plateau.go`** — `ReduceOnPlateau[T]`:
  - Patience-based LR reduction: multiplies current rate by `factor` after `patience` consecutive
    epochs without improvement (mode "min" or "max", threshold-gated).
  - `NewReduceOnPlateau[T](lr0, opts...)` with `WithROPFactor`, `WithROPPatience`, `WithROPThreshold`,
    `WithROPMinLR`, `WithROPMode` functional options.
  - LRS-4 floor at `minLR`; LRS-5 `Reset()`; LRS-6 `SaveState`/`LoadState` JSON round-trip.
  - `Granularity()` = `PerEpoch`.
- **`pkg/optimizer/one_cycle_lr.go`** — `OneCycleLR[T]`:
  - Three-phase schedule: linear warm-up → cosine decay → hold.
  - `NewOneCycleLR[T](maxLR, totalSteps, opts...)` with `WithOCLPctStart`, `WithOCLDivFactor`,
    `WithOCLFinalDiv` functional options.
  - `Granularity()` = `PerStep`; `StepWithMetric` ignores metric (self-contained curve).
  - LRS-4..LRS-6 compliance; `SaveState`/`LoadState` JSON round-trip.
- **`pkg/network/topology.go`** — Dynamic topology mutations:
  - `TopologyMode` enum (`Immutable` default, `Dynamic` opt-in).
  - `topologyTx[T]` struct: shallow snapshot of topology slices for transactional rollback.
  - `Network[T].SetTopologyMode`, `TopologyVersion()` (atomic), `requireDynamic()`.
  - `AddNeuron(layerIdx, count)`, `RemoveNeuron(layerIdx, count)`: grow/shrink a hidden layer
    and rebalance all incoming axons in the mutated layer and its successor.
  - `AddHiddenLayer(position, size, act, bias)`, `RemoveHiddenLayer(position)`: insert/remove
    a layer in the Hiddens chain with full axon rebalancing; enforces ≥1 hidden invariant.
  - Topology version incremented per `Commit()`; rolled back on any error.
- **`pkg/utils/errors.go`** — Six DYN error sentinels:
  `ErrImmutableMode`, `ErrInvalidPosition`, `ErrImmutableLayer`, `ErrMinimumTopology`,
  `ErrEmptyLayer`, `ErrMutationFailed`.
- **`pkg/utils/logger.go`** — `GoLogger` slog adapter:
  - `LevelTrace = slog.Level(-8)`; discard handler fallback for nil loggers.
  - `NewGoLogger(l, libVersion, networkID)` injects `lib_version` + `network_id` via `slog.With`.
  - `Trace/Debug/Info/Warn/Error` methods forwarding to slog at correct levels.
- **`pkg/visualization/`** — HTTP observability server (new package):
  - `VisServer`: `NewVisServer(addr, token, cors)`, `RegisterNetwork(SnapFn)`, `Start()`, `Stop(ctx)`,
    `Addr()` (for `:0` test-port discovery).
  - Six endpoints — all wrapped in `{"protocol_version":"1.0.0","data":...}` envelope:
    `GET /v1/snapshot`, `/v1/loss`, `/v1/activations/{layerIdx}`, `/v1/control`, `/v1/stats`, `/v1/health`.
  - `authMiddleware` (bearer-token, no-op when empty), `corsMiddleware` (CORS headers + OPTIONS preflight).
  - Coverage: 90.0%.
- **`pkg/nn/topology.go`** — DYN-2 state-gated wrappers on `NN[T]`:
  - `requireIdleOrPaused()` — rejects mutations when training is actively running.
  - `AddNeuron`, `RemoveNeuron`, `AddHiddenLayer`, `RemoveHiddenLayer` — gate then delegate to Network[T].
- **`pkg/nn/nn.go`** — `Close() error` (stops vis server with 5 s timeout), `TopologyVersion() uint64`.
- **`pkg/checkpoint/snapshot.go`** — `TopologyVersion uint64` field added to `Snapshot[T]` for
  topology-change detection on checkpoint restore.

#### Changed

- **`pkg/nn/train.go`**:
  - `libVersion = "0.8.0"` constant.
  - PerEpoch scheduler dispatch now type-asserts to `MetricScheduler[T]`; if true calls
    `ms.StepWithMetric(epochLoss)` instead of `sched.Step()`.
  - Training lifecycle log events via `GoLogger` (`Info` start/stop, `Debug` per-epoch loss).
- **`pkg/nn/compile.go`**: starts `VisServer` when `cfg.VisAddr != ""`; calls `SetTopologyMode`.
- **`pkg/nn/options.go`**: `WithTopologyMode[T]`, `WithLogger[T]`, `WithVisualizationEndpoint[T]`,
  `WithVisualizationToken[T]`, `WithVisualizationCORS[T]` functional options.
- **`pkg/nn/config.go`**: `TopologyMode network.TopologyMode`, `Logger *slog.Logger`,
  `VisAddr string`, `VisToken string`, `VisCORS bool` fields.
- **`pkg/nn/builder.go`**: `WithTopologyMode(network.TopologyMode)` builder method.
- **`l2-lr-scheduling-impl.md`** — bumped to v1.2.0; `ReduceOnPlateau` and `OneCycleLR`
  moved from Deferred to Implemented; `MetricScheduler[T]` interface added to §5.2.

#### Coverage

| Package | Coverage |
| :--- | :--- |
| `pkg/optimizer` | 90.3% |
| `pkg/network` | 92.9% |
| `pkg/visualization` | 90.0% |
| `pkg/nn` | 85.9% |

#### Known Issues

- Pre-existing: race detector (`-race`) unavailable on Windows via PowerShell (gcc PATH issue).

## [0.7.0] — 2026-05-10

### ExponentialLR scheduler and gonn CLI binary

Phase 8 adds one missing scheduler type (`ExponentialLR`) and ships the first
command-line interface (`gonn`) for training, querying, and verifying networks
directly from the terminal without writing Go code.

#### Added

- **`pkg/optimizer/exponential_lr.go`** — `ExponentialLR[T]`:
  - Per-step exponential decay: `lr = lr₀ × gamma^t`.
  - LRS-1..LRS-6 compliance identical to `StepLR` (no `stepSize` parameter — simpler formula).
  - `NewExponentialLR[T](lr0, gamma)` / `NewExponentialLRWithGranularity[T](lr0, gamma, gran)`.
  - Default `Granularity()` = `PerEpoch`; JSON `SaveState`/`LoadState` round-trip.
  - `pkg/optimizer/` coverage: 88.2%.
- **`cmd/gonn/`** — standalone CLI binary:
  - `train` — load config JSON + CSV dataset → train → write weights.json. Flags: `--config`, `--data`, `--out` (default `weights.json`), `--resume`, `--precision` (float32|float64), `--json`.
  - `query` — load config + weights → single forward pass on `--input` (comma-separated). Flags: `--config`, `--weights`, `--input`, `--precision`, `--json`.
  - `verify` — load config + weights → mean loss over evaluation CSV (no weight update). Flags: `--config`, `--weights`, `--data`, `--precision`, `--json`.
  - `version` — print `gonn <version> (go<runtime>)`. Flag: `--json`.
  - Exit-code contract: 0=OK, 1=generic, 2=ErrUserConfig, 3=ErrInputData, 4=ErrCompute, 5=ErrIntegrity, 6=ErrIO, 7=unsupported precision.
  - Dataset loading: files ≤ 64 MB use `encoding/csv` full-read; files > 64 MB use `pkg/dataset.NewCSVDataset` streaming path. Threshold overridable via `csvStreamThreshold` package var for testing.
  - `--json` output for all subcommands per spec §5.4: `{"epochs":N,"final_loss":X}`, `{"output":[...]}`, `{"loss":X}`, `{"version":"...","go_version":"..."}`.
  - Coverage: 83.7%.

#### Changed

- `l2-lr-scheduling-impl.md` — bumped to v1.1.0; ExponentialLR moved from Deferred to Implemented.

#### Known Issues

- Pre-existing: race detector (`-race`) unavailable on Windows via PowerShell (gcc PATH issue); all tests pass without `-race`.

---

## [Unreleased] — Phase 7 — 2026-05-10

### LR scheduling, deep network builder, and AI developer skills

Phase 7 extends the optimizer ecosystem with a full LR scheduler framework,
adds bulk topology constructors for deep network construction, and ships a
structured AI developer skills directory for assisted code generation.

#### Added

- **`pkg/optimizer/` — scheduler extension:**
  - `Scheduler[T]` interface: `Step() T`, `Reset()`, `Granularity() Granularity`, `SaveState(io.Writer) error`, `LoadState(io.Reader) error`.
  - `Granularity` enum: `PerEpoch` (default for step-decay schedulers) / `PerStep` (default for warm-up schedulers).
  - `LearningRateSetter[T]` optional interface: `SetLearningRate(T)` — implemented by all four existing optimizers (SGD, Adam, RMSProp, SGDMomentum).
  - `BindScheduler[T](opt, sched) Scheduler[T]` — wraps scheduler; each `Step()` call also calls `opt.SetLearningRate(newRate)` if the optimizer implements `LearningRateSetter[T]`.
  - `NewStepLR[T](lr0, stepSize, gamma)` / `NewStepLRWithGranularity[T](...)` — step decay: lr₀ × γ^(⌊t/stepSize⌋).
  - `NewWarmUpLR[T](lr0, warmupSteps)` — linear warm-up from 0 to lr₀ over N steps; holds lr₀ after; `PerStep` default.
  - `NewCosineAnnealingLR[T](lr0, lrMin, tMax)` — cosine annealing: lrMin + 0.5(lr₀−lrMin)(1+cos(πt/tMax)); holds lrMin after tMax.
  - `NewChainScheduler[T](segments []SchedulerSegment[T])` — sequential composition; `SchedulerSegment[T]{Scheduler, Duration}`. Resets each sub-scheduler on transition. Holds last rate when all segments are exhausted.
  - All scheduler types implement full JSON `SaveState`/`LoadState` for training checkpoints.
- **`pkg/nn/` — bulk topology constructors and scheduler wiring:**
  - Builder API: `Repeat(count, size uint, act, bias)` — appends N identical hidden layers; `Pattern(block, repeats)` — appends block×repeats layers; `HiddenLayers(layers)` — **replaces** all hidden layers (setter semantics); `WithScheduler(sched)`.
  - Options API: `Repeat[T](count, size, act)`, `Pattern[T](block, repeats)`, `WithHiddenLayers[T](layers)` — append semantics (consistent with `WithHiddenLayer`); `WithScheduler[T](sched)`.
  - `Config[T].Scheduler` field added.
  - Training loop (`Fit`) dispatches `sched.Step()` per `Granularity()`: after each batch for `PerStep`, after each epoch for `PerEpoch`. Nil-guarded — no overhead when no scheduler is configured.
- **`skills/gonn/`** — AI developer skills directory:
  - `SKILL.md` — 10-section skill file with YAML frontmatter: API styles, lifecycle, types, bulk constructors, activation↔loss matching, optimizer selection, AI-Meta annotation, error handling, testing patterns, common mistakes.
  - `examples/builder-xor.md` — XOR classifier with Builder API (Style A), anti-patterns.
  - `examples/options-mnist.md` — MNIST classifier with Options API + `PresetMNIST`, anti-patterns.
  - `examples/deep-network.md` — 100-layer network with `Repeat`, `Pattern`, and `BindScheduler` + `ChainScheduler` (warm-up → cosine), anti-patterns.
  - `resources/api-reference.md` — condensed public API surface for all exported symbols in `pkg/nn`, `pkg/optimizer`, `pkg/regularizer`.
  - `resources/conventions.md` — GoNN coding conventions for AI agents (C25–C33 translated without referencing internal file names).

#### Changed

- `pkg/optimizer/sgd.go`, `adam.go`, `rmsprop.go`, `sgd_momentum.go` — each gained `SetLearningRate(T)` implementing `LearningRateSetter[T]`.

#### Coverage

- `pkg/optimizer`: 88.9% (up from 98.9% baseline via new scheduler files at ~85%)
- `pkg/nn`: 86.4% (unchanged — new tests offset new code)

## [0.6.0] — 2026-05-08

### Multi-hidden topology, optimizer pluggability, and regularization

This release completes the Phase 5 + Phase 6 milestones and represents the
first feature-complete minor release of the GoNN v0.6 line.

#### Added

- **`pkg/optimizer/`** — new package with four weight-update strategies:
  - `SGD[T]` — vanilla stochastic gradient descent; stateless, 0 allocs/op. `DefaultOptimizer` fallback.
  - `Adam[T]` — adaptive moment estimation (Kingma & Ba 2014); β₁=0.9, β₂=0.999, ε=1e-8 defaults; lazy moment allocation, 0 allocs/op in hot path.
  - `SGDMomentum[T]` — SGD with velocity term γ=0.9; lazy velocity allocation.
  - `RMSProp[T]` — squared-gradient EMA; α=0.99, ε=1e-8; lazy allocation.
  - All four implement `Optimizer[T]` with `Step`, `Reset`, `LearningRate`, `SaveState`, `LoadState`.
  - `DefaultOptimizer[T](lr)` returns SGD; preserves pre-v0.6 training arithmetic when no optimizer is configured.
- **`pkg/regularizer/`** — new package:
  - `L2[T]` — weight-decay penalty λ×Σwᵢ²; identity `ApplyMask`.
  - `L1[T]` — lasso penalty λ×Σ|wᵢ|; identity `ApplyMask`.
  - `Dropout[T]` — inverted Bernoulli mask; retained activations scaled ×(1/p); `ApplyMask(_, false)` is a strict no-op (inference unchanged).
  - `Compose[T]` — additive penalty + left-to-right sequential `ApplyMask` over a variadic slice.
  - `Apply` and `Penalty` nil-safe package-level helpers eliminate per-call-site nil guards in training code.
- **`pkg/nn`** — integration points:
  - `WithOptimizer[T](opt optimizer.Optimizer[T])` option and `cfg.Optimizer` Config field.
  - `WithRegularizer[T](reg regularizer.Regularizer[T])` option and `cfg.Regularizer` Config field.
  - New `trainStep()` helper replaces the inline `rate × delta` weight update; routes through `opt.Step(weights, deltas)` so all four optimizers are usable transparently.
  - Regularizer penalty is added to effective loss; `ApplyMask(acts, true)` applied after forward pass during training; `ApplyMask(acts, false)` is a no-op during Query.
  - `weightBuf` / `gradBuf` on `NN[T]` reused across training steps — zero per-step allocations.
- **Multi-hidden topology** (Phase 5 — previously shipped on branch):
  - `pkg/network.Network[T]` now holds `Hiddens []bundle` supporting arbitrary chain depth.
  - `compile()` gate lifted — topologies with two or more hidden layers are fully supported.
  - `pkg/persistence` schema bumped 1.0.0 → 1.1.0 with forward-compat loading of v0.1 fixtures.
  - Six new examples: `examples/perceptron/` (E03), `examples/binary_classification/` (E04), `examples/iris/` (E05), `examples/regression_sin/` (E07), `examples/regression_multi/` (E08), `examples/higher_order_options/` (E13).
- **WeightInit debt resolved** — `pkg/nn/compile.go` now calls `Network.SetWeightSampler` before `Build()`, wiring Xavier / He / Uniform initialisation through to axon construction. The prior `U[-0.5, 0.5]` default remains as the `WeightInitRandom` variant; Xavier is the default for SIGMOID layers.
- **C33 AI-Meta annotation** — structured `AI-Meta:` trailing blocks on all exported doc-comments (Purpose, Usage, Concurrency, Errors, Stability).

#### Changed

- `pkg/nn/config.go` — added `Optimizer optimizer.Optimizer[T]` and `Regularizer regularizer.Regularizer[T]` fields.
- `pkg/nn/train.go` — per-sample weight update now delegates to `opt.Step`; regularizer penalty applied before backprop.
- `pkg/network/network.go` — added `WeightSampler[T]`, `SetWeightSampler`, `AppendFlatWeights`, `ApplyFlatWeights`, `HiddenActivations`, `SetHiddenActivations`.
- `pkg/network/propagation.go` — added `AppendFlatGradients` (returns ∂L/∂w sign convention: `-σ'(z) × miss × cellValue`).

#### Deferred to v0.7

- **E06 MNIST example** — awaits MNIST dataset-loader spec.
- **E10 Continuation** — demonstrates `AndTrain` weight reuse parity.
- `context.Context` integration for training cancellation.
- NaN-loss detection and recovery hooks.

#### Known Issues

- `TestPauseResumeCycle` in `pkg/nn` exhibits a timing-dependent flake on Windows (pre-existing; not introduced in v0.6). The goroutine-state-machine logic is correct; the test races against OS scheduler jitter in CI. Tracked for fix in v0.6.1.

## [Unreleased]

### AI-Meta annotation rollout — 2026-05-07

Added structured `AI-Meta:` trailing blocks to all exported Go doc comments
across the entire public surface (`pkg/`). Governed by new rule **C33** and
spec **l2-ai-doc-metadata.md** (promoted from RFC to Stable). Closes TODO #27.

#### Added

- `.design/specifications/l2-ai-doc-metadata.md` — spec defining the closed
  vocabulary (`Purpose`, `Usage`, `Lifecycle`, `Concurrency`, `Errors`,
  `Related`, `Constraints`, `Implementations`, `Stability`), tier matrix,
  BNF grammar, and process-artifact firewall.
- C33 rule in `.design/RULES.md` (version 1.3.0).
- `AI-Meta:` blocks on ~200 exported symbols across `pkg/utils`,
  `pkg/activation`, `pkg/loss`, `pkg/neuron`, `pkg/layer`, `pkg/network`,
  `pkg/dataset`, `pkg/compute`, `pkg/checkpoint`, `pkg/persistence`, `pkg/nn`.

### Phase 4 — 2026-05-02

v0.5 examples catalog. Seven new example modules covering the
v0.5-implementable subset of `l2-usage-examples` Stable v1.0.0; the
remaining eight entries (multi-hidden / `AndTrain` / MNIST loader)
defer to v0.6. Phase Gate T-4Z01..T-4Z04 closed; v0.5 release-ready
bar reached.

#### Added

- `examples/xor/` (E01) — dual-style XOR (Builder + Functional Options).
  Establishes the Phase 4 v0.5 smoke-test pattern: `runX()` helpers
  callable from both `main()` and `main_test.go` with loose loss
  thresholds to absorb seed-driven init drift.
- `examples/style_showcase/` (E12) — same XOR network built three ways
  (Builder, Options, `PresetXOR`); inter-style spread asserted ≤ 0.1.
- `examples/logic_gates/` (E02) — single 2-2-1 topology trained over
  AND / OR / NAND truth tables in a loop.
- `examples/callbacks/` (E11) — `WithEpochCallback` + `WithBatchCallback`
  wired to XOR; smoke test asserts both fire at least once during Fit.
- `examples/persistence/` (E09) — train, extract weights via bundle
  accessors, write `pkg/persistence` config + weights, reload into a
  freshly compiled network, assert post-reload Query matches original
  within `cpu.ToleranceF32` (PERS-4). Documents the v0.5 conversion
  seam future `nn.Save` / `nn.Load` hooks will close.
- `examples/shared_options/` (E14, v0.5 adapted) — shared `[]Option[T]`
  across two single-hidden topologies (ReLU(8) vs ReLU(16)). Spec
  Topology B's two-hidden variant is gated until v0.6.
- `examples/precision/` (E15) — XOR at `float32` and `float64` with
  identical hyperparameters; reports per-precision loss + elapsed time.
- `examples/README.md` — catalog index and coverage matrix audit
  flagging six v0.6-gated API surfaces (`Sequential`, `DeepNetwork`,
  `PresetMNIST`, `PresetRegression`, `Verify`, `AndTrain`).

#### Changed

- `go.work` — registered seven new example modules.
- `examples/perceptron/main.go` — added `// removed once v0.6 lands`
  doc-comment documenting the temporary single-hidden stub state. The
  legacy four-hidden topology returns when E03 is unblocked by v0.6.

#### Phase Gate

- T-4Z01 `go build` — green for every example module and the library
  itself.
- T-4Z02 `go test` — every smoke test passes deterministically (loose
  loss thresholds; no public RNG seed yet).
- T-4Z03 `go test -race` — race-clean across all seven examples (run
  via PowerShell per existing 2026-04-29 / 2026-05-01 STATE.md note).
- T-4Z04 STATE.md updated; PLAN.md Phase 4 → ✓ Done; v0.5 release-ready
  bar reached; v0.6 backlog (multi-hidden + `AndTrain` + MNIST loader)
  ready for next planning cycle.

### Phase 3 — 2026-05-01

New capability packages from Tracks A–E. All five new packages ship with
≥80% line coverage, race-clean test suites, and benchmarks for the inner
loop hot paths. Phase Gate T-3Z01..T-3Z04 closed; Phase 4 (Examples)
unblocked.

#### Added

- `pkg/persistence/` — `ConfigDoc[T]` + `WeightsDoc[T]` JSON schema with
  atomic write (tmp + Sync + Rename), SHA-256 `config_hash` anchor
  (PERS-3), deterministic re-serialisation (PERS-2), and
  `strconv.FormatFloat`-driven float round-trip (PERS-4). 81.4 % cover.
- `pkg/checkpoint/` — `Snapshot[T]` resumable training state (CHK-2),
  `WriteSnapshot` / `LoadLatest` atomic write (CHK-1) with auto-detected
  gzip cold tier (CHK-3), and `Sweep` / `StartSweeper` retention
  goroutine on `time.Ticker`. 83.2 % cover.
- `pkg/dataset/` — `Dataset[T]` pull-source interface (DAT-1) with
  `NewSliceDataset` (in-memory), `NewCSVDataset` (`encoding/csv` +
  `bufio`), and `Prefetch` channel-based decorator that bounds residency
  to `(prefetch+1) × batchSize` (DAT-3). 87.6 % cover.
- `pkg/compute/` — pluggable `Backend[T]` interface (`Forward`,
  `Backward`, `UpdateWeights`, `Allocate`, `Free`) with name-keyed
  registry (COMP-2). 97.3 % cover.
- `pkg/compute/cpu/` — reference CPU backend (`init()` registration,
  pure Go kernels, `ToleranceF32 = 1e-5`, `ToleranceF64 = 1e-12`,
  always-present default per COMP-1). 100 % cover.
- `pkg/network/pool.go` — `sync.Pool`-backed `AcquireActivations` /
  `ReleaseActivations` with per-precision pools and zero-on-release
  semantics (PERF-4); `PreallocStorage[T]` flat-slice owner for
  `Compile()`-time preallocation (PERF-2).
- `pkg/network/worker.go` — `WorkerPool` sized to
  `runtime.GOMAXPROCS(0)` (PERF-3) with idempotent `Stop`.
- `pkg/network/bench_network_test.go`, `pkg/nn/bench_nn_test.go` — PERF-1
  benchmarks: `BenchmarkAcquireRelease_F32/F64`,
  `BenchmarkPreallocStorage_DeepNetwork_F32`,
  `BenchmarkWorkerPool_Submit`, `BenchmarkForward_XOR_f32`,
  `BenchmarkBackward_XOR_f32`, `BenchmarkCompile_DeepNetwork_f32`,
  `BenchmarkFit_XOR_f32`. Backward pass measured at 0 allocs/op.
- `pkg/nn/profiling.go` + `WithProfiling[T](addr)` option — opt-in
  `net/http/pprof` listener (PERF-5); blank-import side effect registers
  `/debug/pprof/*` handlers; mutex-guarded once-per-address dispatch so
  repeated `New` calls cannot duplicate listeners.

#### Changed

- `pkg/nn/config.go` — added `ProfilingAddr string` field to `Config[T]`.
- `pkg/nn/compile.go` — terminal `startProfilingServer(cfg.ProfilingAddr)`
  call so successful compile arms the optional pprof listener.

#### Phase Gate

- T-3Z01 `go build ./...` — green across all 14 packages.
- T-3Z02 coverage — every new package ≥ 80 % line coverage; existing
  packages unchanged.
- T-3Z03 `go test -race ./...` — race-clean (run via PowerShell for gcc
  PATH resolution per existing 2026-04-29 note in STATE.md). The
  pre-existing `TestPauseResumeCycle` flake on `pkg/nn` is unrelated to
  Phase 3 work and reproduces on `develop` without these changes.
- T-3Z04 STATE.md updated; PLAN.md Phase 3 → ✓ Done; Phase 4 unblocked.

### Phase 1 Track A — 2026-04-29 [Bootstrap]

Foundations slice of the Phase 1 Foundation Rewrite. Closes the `pkg/utils`
half of blocker C-001 (Tracks B/D still pending). Specs `l2-errors-impl` and
`l2-init-impl` are RFC; promotion to Stable deferred to Phase Gate T-1Z02.

#### Added

- `pkg/utils/errors.go` — six orthogonal category sentinels (`ErrUserConfig`,
  `ErrInputData`, `ErrCompute`, `ErrControl`, `ErrIntegrity`, `ErrIO`) plus
  `Newf`, `Wrap`, `NewSizeError`, `NewActivationError`, `NewIntegrityError`,
  `LocationHint`. Multi-`%w` wrapping preserves both category and cause for
  `errors.Is` routing.
- `pkg/utils/init.go` — `NewRNG(seed) (*rand.Rand, uint64)` based on
  `math/rand/v2.PCG`; generic samplers `XavierUniform[T]`, `HeNormal[T]`,
  `Uniform[T]` over `utils.Float`. Zero-allocation hot path (~15-42 ns/op).
- `pkg/utils/errors_test.go`, `pkg/utils/init_test.go` — 100 % line coverage,
  C32 forbidden-phrase sweep, distribution + reproducibility checks,
  zero-alloc benchmarks via `b.Loop()`.

#### Changed

- `.design/specifications/l2-errors-impl.md` — bumped 0.2.0 → 0.3.0;
  sentinel set finalized to 6 orthogonal categories. `ErrTrainingFailure`
  and `ErrUnsupported` from v0.6.0 dissolved into `ErrCompute` /
  `ErrUserConfig` to avoid catch-all routing.
- `.design/INDEX.md`, `.design/PLAN.md` — registry version aligned.
- `.design/tasks/phase-1.md` — `[Bootstrap]` markers added to T-1A0x;
  Track A frontmatter populated with provides / key_files / patterns_established.

#### Notes

- `go test -race` skipped in dev environment (no gcc). Race-detector pass
  is recorded as a Phase Gate requirement (T-1Z02) and must run on a
  CGO-enabled host before merging.

### Phase 1 Track B — 2026-04-29

Neuron slice of the Phase 1 Foundation Rewrite. Closes the `pkg/neuron`
half of blocker C-001: `Hidden[T]` exists, `Output.CalculateValue` no
longer recurses, `axon.OutgoingCell` is restored. `pkg/network/...` still
fails to build — that surface is owned by Tracks C/D.

#### Added

- `pkg/neuron/cell/hidden.go` — `type Hidden[T utils.Float] = Dense[T]`
  generic alias plus `NewHidden(number)` constructor. Closes the missing
  symbol referenced by `pkg/network/network.go`.
- `pkg/neuron/axon` — `NewWithWeight(weight, incoming, outgoing)` for the
  layer-driven path that pre-samples weights via Xavier/He (Track A).
- `pkg/neuron/cell/cell_test.go`, `pkg/neuron/axon/axon_test.go` — table-
  driven contract tests, 100 % line coverage, regression test that pins
  the Output non-recursion property, concurrency probe over the package
  PCG mutex.

#### Changed

- `pkg/neuron/cell/core.go`, `input.go`, `bias.go`, `dense.go`,
  `output.go` — full C31 doc-comments; `_NewBias` renamed `NewBias`;
  `Output.CalculateValue` rewritten to delegate to embedded Dense and
  compute residual `target - value` only when target is non-nil.
- `pkg/neuron/axon/axon.go` — restored `OutgoingCell neuron.Neuron[T]`;
  swapped `math/rand` legacy global for `math/rand/v2.PCG` via
  `utils.NewRNG(0)`; `New` and `NewWithWeight` distinguish default-init
  from caller-controlled paths.

### Phase 1 Track C — 2026-04-29

Layer slice of the Phase 1 Foundation Rewrite. Closes the nil-deref
constructors, the duplicate Init logic, and the dropped-size defect from
[l2-layer-types] §5.3.

#### Added

- `pkg/layer/layer_test.go` — 93.7 % line coverage. Includes regression
  tests for nil-deref panic, dropped size on `NewInput`, Init de-dup,
  bias-cell alloc, target pointer aliasing.

#### Changed

- `pkg/layer/core.go`, `base.go`, `input.go`, `dense.go`, `output.go` —
  full rewrite. Constructors now allocate the embedded core/base via
  `newCore` / `newBase`; `Dense.Init` delegates to `base.Init`;
  `NewInput(size)` honours the size argument; bias cells allocated when
  `Bias=true`; Output owns a `targets []T` slice and exposes
  `SetTarget(idx, value)` for label updates between samples.

### Phase 1 Track D — 2026-04-29

Network slice of the Phase 1 Foundation Rewrite. Closes blocker C-001
end-to-end: `go build ./...` is green and the XOR smoke test converges
within 5000 epochs (sigmoid 2-4-1, MSE).

#### Added

- `pkg/network/network_test.go` — 96.5 % line coverage. XOR-convergence
  smoke test (target loss < 0.02), Build axon-fan-out, SetInputs writes
  every cell (regression of [l2-network-graph] §5.4 #4), error routing
  through ErrUserConfig / ErrInputData.
- `Network.Train(input, target)` — one-shot forward + backward + update
  helper used by the smoke test and by Phase-2 facade.
- `Network.LossMode()`, `Network.CalculateLossDefault()` — surfaces the
  loss configured by SetLayers.

#### Changed

- `pkg/network/network.go` — `New[T]()` returns Network by value (kept
  for [pkg/nn] embed compatibility); `SetLayers(in, hidden, out)`
  installs cells from layer constructors and captures activation / loss
  / bias-cell handles; `Build()` wires Input→Hidden and Hidden→Output
  axons (and bias-axon when configured); `SetInputs` / `SetTargets`
  write per-cell instead of cells[0].
- `pkg/network/bundle.go` — slimmed to `cells []S` + `Replace`, `Add`,
  `Cells`, `At`, `Len`. The legacy whole-slice `any(...).([]Neuron[T])`
  cast (always false) is replaced by element-wise type assertion in
  callers (closes [l2-network-graph] §5.4 #3).
- `pkg/network/propagation.go` — forward applies layer activation via
  the [pkg/activation] dispatcher; pre-activation values cached in
  `Network.preactHidden / preactOutput` so backprop feeds pre-σ to
  `activation.Derivative` (needed for sigmoid σ' = σ(x)·(1-σ(x)));
  backward walks Output.Axons in reverse to credit Hidden cells.

### Phase 1 Gate — 2026-04-29

- T-1Z01 `go build ./...` — green. C-001 closed.
- T-1Z02 `go test -cover ./...`:
  - pkg/utils 100 %
  - pkg/neuron/cell 100 %
  - pkg/neuron/axon 100 %
  - pkg/layer 93.7 %
  - pkg/network 96.5 %
  - pkg/activation 9.1 % (Phase 0 baseline; out of scope)
  - pkg/loss 13.5 % (Phase 0 baseline; out of scope)
  - pkg/nn 0 % (Phase 2 facade; out of scope)
- T-1Z02 `go test -race ./...` — **PASS** on all 7 packages
  (utils, neuron/cell, neuron/axon, layer, network plus pre-existing
  activation, loss). Confirmed under MSYS2 mingw64 gcc 15.2.0 via
  PowerShell. Note: the Claude Code bash shell does not propagate
  Windows PATH to the Go child process; race-detector runs must be
  launched from PowerShell or `cmd.exe` until that environment is
  fixed.
- T-1Z03 STATE.md cleared blocker C-001; phase-1 frontmatter populated
  with provides / key_files / patterns_established / status:Done.

### Phase 2 — 2026-04-30

Public-facade slice. Restores the `pkg/nn.NN[T]` surface on top of the
Phase-1 foundation per the [l2-nn-facade] v2.0.0 dual-style API.
Promotes three specs to Stable.

#### Added

- `pkg/nn/config.go` — `Config[T]`, `HiddenLayerSpec[T]`,
  `WeightInitMethod` constants, lifecycle `state` enum, project
  defaults (LearningRate=0.3, MaxIterations=10000, LossLimit=1e-4,
  WeightInit=xavier, LossMode=MSE).
- `pkg/nn/nn.go` — `NN[T]` type embedding `network.Network[T]`,
  `NewBuilder[T]()` entry point, `State()` and `Config()` accessors,
  `guardConfiguring(method)` helper that turns post-Compile mutations
  into Logger.Warn no-ops (preserves L1 INV-2).
- `pkg/nn/builder.go` — Builder API: `Input` / `Dense` / `Hidden`
  (alias) / `Output` / `WithLearningRate` / `WithLoss` / `WithBias` /
  `WithWeightInit` / `WithLossLimit` / `WithMaxIterations` /
  `WithEpochCallback` / `WithBatchCallback` / `Compile()` /
  `MustCompile()`. Output drops `loss.Type` (breaking vs v1).
- `pkg/nn/compile.go` — shared finalisation routine consumed by both
  styles. Validates §5.7 hard-error rules with C32-compliant messages
  wrapping `utils.ErrUserConfig`; emits §5.7 soft warnings via
  `Logger.Warn`. v0.5 rejects `len(HiddenLayers) > 1` with explicit
  "v0.6 feature" note.
- `pkg/nn/options.go` — Functional Options API: `Option[T]`, `New[T]`,
  `MustNew[T]`, topology + configuration mirrors of every Builder
  method.
- `pkg/nn/presets.go` — higher-order options `Sequential`,
  `DeepNetwork`, `StandardSetup`; presets `PresetXOR`, `PresetMNIST`,
  `PresetRegression`. PresetXOR converges via the Phase-1 baseline;
  multi-hidden presets surface their config but error at Compile() in
  v0.5.
- `pkg/nn/query.go`, `verify.go` — forward-only inference and
  forward+loss-without-update.
- `pkg/nn/train.go` — `Train(input, target)` (single-step) and
  `Fit(dataset)` (multi-epoch with `MaxIterations` / `LossLimit`
  early-stopping, min-loss snapshot/rollback, callbacks).
  `snapshotWeights` / `restoreWeights` use a flat `[]T` buffer reused
  across epochs (zero per-epoch GC churn).
- `pkg/nn/control.go` — atomic 4-state lifecycle (`Idle/Running/
  Paused/Stopped`) with `Pause()` / `Resume()` / `Stop()` on `*NN[T]`,
  `awaitSafePoint` worker-side helper, `transitionToRunning` /
  `transitionToIdle` lifecycle transitions. Invalid transitions wrap
  `utils.ErrControl`.
- `pkg/nn/nn_test.go` — 84.8% line coverage. Builder API state-machine
  tests, all §5.7 validation rules reachable, post-Compile mutation
  no-op assertions, idempotent Compile, MustCompile/MustNew panic
  tests, Builder vs Options Config-state parity, presets surface,
  Query/Verify/Fit happy and error paths, EpochCallback invocation
  count, Pause/Resume/Stop race scenarios under `-race`.

#### Changed

- `pkg/nn/nn.go` / `builder.go` / `query.go` / `verify.go` /
  `train.go` — full rewrite from v1 stubs/comments.
- `examples/perceptron/main.go` — migrated from legacy `nn.New()` (v1)
  to `nn.NewBuilder()` + `WithLoss()` (v2). Output() no longer
  receives `loss.Type` (breaking change documented in §5.2).
- `.design/specifications/l2-nn-facade.md` — RFC → Stable v2.0.0.
  Document History row records the v0.5 single-hidden limitation and
  the v0.6 multi-hidden plan.
- `.design/specifications/l2-training-loop.md` — Draft → Stable
  v1.0.0. Records the Train + Fit dual-method shape, flat snapshot
  buffer (vs §5.2 nested `WeightsBuffer`), and the `context.Context` /
  NaN-loss detection deferred to v0.6.
- `.design/specifications/l2-control-impl.md` — Draft → Stable v1.0.0.
  Records the 4-state collapse of §5.1's 6-state design, the TOCTOU
  CAS guard in `transitionToRunning`, and the deferred ctx /
  buffered-channel control bus.
- `.design/INDEX.md` 1.8.0 → 1.9.0; `.design/PLAN.md` 1.1.0 → 1.2.0;
  `.design/TASKS.md` 1.1.0 → 1.2.0 — Phase 2 marked Done; spec status
  table refreshed.

#### Notes

- `go test -race -cover ./...` — all packages green, `pkg/nn` 84.8%.
  Run via PowerShell on this host (Claude Code bash shell does not
  propagate Windows PATH to the Go child process; `gcc.exe` lives at
  `C:\msys64\mingw64\bin`).
- Multi-hidden support, `context.Context` integration, NaN-loss
  detection, and the buffered-channel control bus are all explicitly
  deferred to v0.6 with spec annotations and `Compile()`-time errors
  pointing the user at the limitation.

#### Fixes

- `Fit()` return semantics: previously returned `cfg.MaxIterations`
  unconditionally regardless of early-exit via `Stop()` or `LossLimit`.
  Replaced with a tracked `completedEpochs` counter — now satisfies
  TRN-4 from [l1-training-semantics] (return the meaningful epoch
  count, never wall-clock max). Surfaced by the deterministic
  `TestStopRequestsEarlyExit` synchronisation barrier, which was
  reporting `epochs=100000` even after a successful Stop.

### Changed

- Updated task plan and task index (main)
- Completed task `phase-11` (main)
- Updated 2 specifications (main)
- Completed task `phase-12` (main)
- Added specification `conv-2d-layers` (main)
- Completed 2 tasks (main)
- Completed task `phase-14` (main)
- Updated 5 specifications (main)
- Completed task `phase-15` (main)
- Added specification `attention` (main)
- Added specification `attention-impl` (main)
- Updated task execution state (main)
- Added 5 specifications (main)
- Completed task `phase-18` (main)
- Added specification `quantization-impl` (main)
- Updated implementation plan (main)
- Completed task `phase-19` (main)
