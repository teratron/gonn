# GoNN Full Restoration Plan

## Progress (execution status)

**Completed and verified** (full suite green, `-race` clean, gradcheck oracle passing, XOR example converges):

- **Phase 0** — safety net: `internal/verification/` with the gradient-check oracle (5 topologies × 8 activations, worst rel err < 1e-4), ~25 scenario/lifecycle/wiring tests, `.github/workflows/ci.yml`.
- **Phase 1** — core math: chain-rule fix in `CalculateMisses` (folds downstream σ′); `tanhDerivative` and SELU-alpha fixes; `loss.Derivative` for all 18 losses + `VectorLoss`/`Aggregate`; real CCE; miss = −∂ℓ/∂y wired into `CalculateValues`; true vector softmax (`SoftmaxInto`); fused (SIGMOID,BCE)/(SOFTMAX,CCE); SOFTMAX+non-CCE and hidden-SOFTMAX rejected at compile. MSE uses the ½ convention to keep training numerics stable.
- **Phase 2** — unfroze Conv2D/SimpleRNN/LSTM/GRU/PositionalEncoding via an `ApplyGradSGD` capability interface (replaced the brittle type switch in `train.go`); embedding lazy-init (fixes compile panic); positional-grad reset.
- **Phase 3** — stateless `Query` (`Network.InferDense`) under `RWMutex` (race-free, verified with a 16-goroutine storm under `-race`); `Stop` no longer sterilizes; `AndTrain` LR sync; incremental `rebalance` preserves learned weights; optimizer moment-buffer resize guards (no Adam panic); NaN/Inf input guards; `ApplyFlatWeights` returns error; `Verify` runs the conv prefix; callbacks honor the stop contract; attention shape guard.
- **Phase 4a** — L1/L2 fold into the gradient (`Regularizer.WeightGrad` + `AddWeightGrad`); scheduler auto-binds to the optimizer in compile (idempotent `BindScheduler`).
- **Phase 5a** — constant-time vis token compare; gzip-bomb limit in checkpoint reader; IDX record-length overflow cap; pprof non-loopback warning.
- **Phase 6a** — `libVersion` → 0.11.0; `TestPauseResumeCycle` de-flaked.

**Remaining (deferred — feature-wiring, not correctness of the trained model):**

- Phase 4b: BatchNorm/LayerNorm/GroupNorm in the dense forward path; honest per-layer Dropout (currently still masked post-forward); `nn.Save`/`nn.Load` public API; checkpoint integration into Fit; real visualization snapshot; quantize dense layers; `FitDataset` bridge + `prefetch.Close`.
- Phase 4c: SoA weight storage + routing the forward/backward through `compute.Backend` (largest, riskiest — deliberately not attempted to protect the now-correct core).
- Phase 5b: pprof graceful shutdown / dedicated mux; vis HTTP timeouts.
- Phase 6b: remaining dead-code removal (`topologyTx`, attention placeholder, `bufAttn`), Pause busy-wait → `sync.Cond`, full README/CHANGELOG truth pass.

These deferred items are "feature is inert" (a user not using BatchNorm/Dropout/Save/GPU is unaffected); the correctness-critical findings (audit blocks A, C, D, plus L2/scheduler from B and the quick wins from E) are all fixed and regression-guarded.

## Context

A two-pass engineering audit (static review of all 16 packages + 25 live simulations with numeric gradient checks) found that GoNN's leaf-level math is largely correct (Conv1D/Conv2D/attention/RNN verified to ~1e-9 against central differences) but the composition layer is broken: backprop drops activation derivatives across layer boundaries, loss selection never reaches gradients, Conv2D/RNN/LSTM/GRU weights are never updated, embeddings panic at compile, BatchNorm/L1/L2/Dropout/scheduler/GPU-backend/visualization are silently disconnected, `Query` races despite a "ReadSafe" contract, and several lifecycle bugs (sterile `Stop`, ignored `AndTrain` LR, Adam panic after topology mutation) corrupt or crash training.

User decisions: **full restoration** (all findings), **stateless Query**, **all 18 loss functions with true derivatives**, **wire up both visualization and the compute backend**. Training numerics will change for every user — this is a deliberate correctness-breaking release (target `v0.11.0`).

The audit's 25-scenario harness lives in the session scratchpad `sim/` module and is the seed for the regression suite.

Guiding principle: every fix lands behind a numeric oracle (gradient check / convergence test) written first. The current green test suite pins broken behavior; affected tests are rewritten to assert the mathematically correct outcome, never deleted silently.

## Phase 0 — Safety net (before any fix)

Create `internal/verification/` test package (build-tag-free, plain `go test`):

1. **Gradient-check oracle** `internal/verification/gradcheck_test.go`: central-difference vs `AppendFlatGradients` for a matrix of topologies — {1,2,3 hidden} × {1,3 outputs} × every activation × representative losses. All FAIL initially (they encode target behavior); they gate every later phase. Reuse the harness logic from the audit's `sim/main.go` `scnGradCheck`.
2. **Port the 25 audit scenarios** as tests with *correct* expectations: `conv2d_frozen`, `rnn_frozen`, `posenc_accum`, `stop_sterile`, `andtrain_lr`, `race_query` (with `-race`), `cce_stop`, `bce_negative`, `softmax_sum`, `nan_guard`, `verify_conv`, `adam_topology`, `sched_bind`, `quant_mlp`, etc.
3. **Convergence goldens**: XOR (sigmoid/tanh/relu hidden), 2-hidden multi-output, one conv1d / conv2d / rnn / attention-prefix network: assert `loss < threshold` within N epochs.
4. CI wiring: `go vet`, `go test ./... -race`, verification suite. (Repo has no CI config — add `.github/workflows/ci.yml`.)

## Phase 1 — Core training math (audit block A)

Files: `pkg/network/propagation.go`, `pkg/activation/*.go`, `pkg/loss/*.go`, `pkg/nn/compile.go`.

1. **Unify derivative convention = pre-activation z** (matches sigmoid/relu/elu/selu/swish/elish already):
   - Fix `tanhDerivative` (`pkg/activation/tanh.go:14`) → `1 − tanh²(z)`.
   - Fix SELU alpha typo in `Derivative` dispatcher (`pkg/activation/activation.go:142`).
   - Correct the `Derivative` doc comment ("post-activation" claim is wrong).
2. **Chain-rule fix in `CalculateMisses`** (`pkg/network/propagation.go:67`): when propagating miss from layer i+1 (or Output) into layer i, multiply by the *source layer's* σ′(preact): output→last-hidden uses `Derivative(preactOutput[oi], outputAct)`; hidden i+1→i uses `Derivative(preactHiddens[i+1][ci], hiddenActs[i+1])`. `CalculateWeights` / `AppendFlatGradients` / `AppendInputGradient` keep their local-σ′ formulas and become exact automatically. Oracle: Phase-0 gradcheck goes green.
3. **Loss derivatives** in `pkg/loss`: add `Derivative[T](predicted, target T, mode Type) T` (element-wise) and `VectorLoss/VectorDerivative` for CCE, COSINE, CAT_HINGE. Implement all 18 (MSE `(y−t)`, MAE/AVG sign, BCE clamped `(y−t)/(y(1−y))`, MSLE, KLD, POISSON, MAPE, HINGE family (document t∈{−1,1}), LOG_COSH `tanh(y−t)`, HUBER clamp, ARCTAN, RMSE=MSE-gradient reporting-only). Delete `cceLossSingle ≡ 0` (`pkg/loss/cce.go`).
4. **Wire loss into the residual**: output miss becomes `−loss.Derivative(y, t, mode)` (sign keeps MSE path bit-compatible with today's `t−y`). Rewrite `CalculateLoss` to feed real `(y, t)` pairs (it currently calls `Loss(0, residual)`); retire the sqrt-hack for MSLE/LOG_COSH/HUBER in `CalculateTotalLoss` (keep sqrt for RMSE only).
5. **True softmax output**: vector path in `CalculateValues` when `outputAct == SOFTMAX` using a shared stable helper (extract `softmaxRowwise` from `pkg/layer/attention/cell.go` into `pkg/activation`). **Fused output gradients** for (SIGMOID,BCE) and (SOFTMAX,CCE): flag on `Network` set in `SetLayers`; when fused, output-layer σ′ is skipped and miss = `t − y` (standard shortcut, numerically stable). SOFTMAX on hidden layers → compile error in `validate`.
6. Fix `emitSoftWarnings` guidance accordingly.

Rewrite pinned tests in `pkg/network`, `pkg/nn`, `pkg/loss`, `pkg/activation` that assert the old numerics.

## Phase 2 — Frozen layers & embeddings (block C)

Files: `pkg/nn/train.go`, `pkg/layer/conv/*.go`, `pkg/layer/recurrent/*.go`, `pkg/layer/embedding/*.go`.

1. **Replace the type-switch in `applyConvBackward` (`pkg/nn/train.go:150-180`) with one capability interface** `interface{ ApplyGradSGD(lr T) }` (already implemented by attention/transformer/FFN/TokenEmbedding). Add `ApplyGradSGD` to: `Conv1D`, `Conv2D` (from existing `gradW/gradB`), `SimpleRNN`, `LSTM`, `GRU` (from `gradWxh/gradWhh/gradBh` & gate equivalents), `PositionalEncoding`. This single change unfreezes Conv2D and all recurrent layers.
2. **Gradient-reset contract**: every layer's `Backward` zeroes its accumulators at entry (most already do). Add missing `clear(pe.gradTable)` in `PositionalEncoding.Backward` (`pkg/layer/embedding/positional.go:101`) — fixes unbounded accumulation.
3. **Lazy-init for embeddings** (fixes compile-time panic): in `TokenEmbedding.ForwardIDs` allocate `lastIDs/lastOut` when nil and return zero rows when `Table` is nil (mirror the lazy pattern in `attention.MultiHeadAttention.Forward` at `multihead.go:222`); same for `PositionalEncoding.Forward`. Swallowed error in `TokenEmbedding.Forward` (`out, _ :=`) → log Warn + zero output.
4. **GradClipNorm parity**: pass prefix-layer gradients through `optimizer.ClipByGlobalNorm` together with dense grads in `trainStep`.
5. Oracle: `conv2d_frozen`/`rnn_frozen` now show |dW| > 0 and convergence; prefix-through gradcheck; `posenc_accum` ratio 1:1:1.

## Phase 3 — Lifecycle, concurrency, robustness (block D)

Files: `pkg/nn/{query.go,control.go,andtrain.go,train.go,verify.go,nn.go}`, `pkg/network/{network.go,topology.go}`, `pkg/optimizer/*.go`, `pkg/layer/attention/multihead.go`, `pkg/nn/compile.go`.

1. **Stateless `Query` + RWMutex**: add `mu sync.RWMutex` to `NN`. `Train/Fit/AndTrain/SetTrain/SetEval/topology mutations` take `Lock`; `Query/Verify` take `RLock` and run a new non-mutating forward: `Network.InferInto(input []T, scratch *InferScratch) []T` reading weights from axons into per-call buffers (reuse `AcquireActivations/ReleaseActivations` from `pkg/network/pool.go` — currently dead code, this is its intended job). Conv prefix inference: add optional `ForwardInference(x []T) []T` (no cache writes) to conv/attention/recurrent layers via type-assert; layers lacking it make that Query take the write lock as fallback. Update concurrency docs.
2. **`Stop` no longer sterilizes**: `transitionToIdle` (`pkg/nn/control.go:95`) always stores `controlIdle`; Fit returns `StopExternalStop` for observers. Fix the misleading `AndTrain` "must be Idle, got 3" path.
3. **`AndTrain` option sync** (`pkg/nn/andtrain.go`): after applying opts, set `n.LearningRate = n.cfg.LearningRate` and push into the optimizer via the existing `LearningRateSetter` interface (`pkg/optimizer/scheduler.go:67`); snapshot & restore both in the defer.
4. **Topology mutations preserve learning** (`pkg/network/topology.go`): rewrite `rebalance` into incremental rewiring — keep existing axons/weights for surviving (src,dst) pairs, sample weights only for new pairs, drop axons whose source was removed (pointer identity). Delete the fake `topologyTx` (dead code). After any successful mutation, facade calls `n.opt.Reset()` and invalidates `weightBuf/gradBuf/snapshot`. Add slice-length guards to `Adam/RMSProp/SGDMomentum.Step` (re-allocate moments + Warn on mismatch) so stale state degrades instead of panicking.
5. **Shape pre-pass for attention/transformer/embedding prefixes** in `pkg/nn/compile.go` (mirror `setupRecurrentShapes`): `InputSize()` mismatch → `ErrUserConfig` instead of index-out-of-range panic escaping `New`. Also add length guards in `MultiHeadAttention.Forward` (return `[]T{}` like conv layers).
6. **NaN/Inf guard**: `SetInputs/SetTargets` reject non-finite values with `ErrInputData`; CSV `parseRow` and MNIST loader likewise.
7. **`ApplyFlatWeights`** returns `error` on length mismatch (silent truncation today at `pkg/network/network.go:345`).
8. **`Verify` runs the conv prefix** (`pkg/nn/verify.go`) — same path as Train/Query.
9. **Callbacks honor their contract** (`pkg/nn/callbacks.go:140`): any non-nil callback error stops training (docs already promise this); log at Warn. Add `FitContext(ctx, samples)` so `StopContextCancel` stops being a dead enum value.

## Phase 4 — Wire up the disconnected subsystems (block B)

1. **Normalization in the dense path**: move `normLayers` ownership into `Network`; `CalculateValues` applies `Forward` after layer-i activation; `CalculateMisses` routes the miss vector through `Backward` when crossing layer i; γ/β update via `ApplyGradSGD` in `trainStep`. Rework `BatchNorm` (`pkg/layer/norm/batchnorm.go`) to per-feature running stats (scalar `runningMean/runningVar` today is wrong even conceptually) with sample-stream EMA semantics documented; LayerNorm/GroupNorm plug in unchanged. Oracle: batchnorm test shows output difference + gradcheck through the norm layer.
2. **L1/L2 reach gradients**: extend `Regularizer` with `WeightGrad(w T) T` (L2: `2λw`, L1: `λ·sign(w)`, Dropout: 0, Compose: sum); `trainStep` adds it into `gradBuf` before `opt.Step`. Oracle: λ>0 visibly shrinks weights.
3. **Honest Dropout in the dense path**: per-layer mask hook — `CalculateValues` applies mask to layer i output *before* layer i+1 consumes it; `CalculateMisses` applies `BackwardMask` (currently only called by tests). Delete the post-forward masking block in `trainStep` (`pkg/nn/train.go:74-79`) and the inference-time mask call in `Query`.
4. **Scheduler autobind** in `compile`: when `cfg.Scheduler` is set and optimizer implements `LearningRateSetter`, wrap via `optimizer.BindScheduler` automatically (idempotence: detect already-bound via an unexported marker method).
5. **`nn.Save` / `nn.Load`**: promote the extract/install logic from `cmd/gonn/load.go:85-211` into `pkg/nn` public API (`Save(configPath, weightsPath)`, `Load[T](configPath, weightsPath) (*NN[T], error)`), covering conv-prefix layers via their existing `MarshalJSON`. CLI and `examples/persistence`, `examples/continuation` switch to it.
6. **Checkpoint integration**: `WithCheckpoint(dir string, everyN uint, cfg checkpoint.SweepConfig)` — Fit writes `checkpoint.WriteSnapshot` every N epochs (weights via the new Save internals) and `nn.Resume(dir)` restores; persist optimizer state via existing `SaveState/LoadState` and RNG via `rand.PCG.MarshalBinary`. Sweeper errors get logged (fix silent `_, _, _, _ =` in `pkg/checkpoint/retention.go:134`).
7. **Visualization becomes real**: Fit publishes an epoch-end snapshot into `atomic.Pointer[visualization.NetworkState]` (loss, epoch, iteration, layer sizes, hidden activations copy); `startVisServer` registers a SnapFn reading that pointer instead of the hardcoded stub (`pkg/nn/compile.go:183`).
8. **Compute backend actually computes** (largest sub-project, do last in this phase):
   - Step 1 — SoA weight storage: per-layer flat weight matrices owned by bundles; `Axon` becomes a view (index pair) kept for API compatibility; `FlatWeights/ApplyFlatWeights` become copies over contiguous storage.
   - Step 2 — route `CalculateValues/CalculateMisses/CalculateWeights` through `compute.Backend.Forward/Backward/UpdateWeights` (`pkg/compute/cpu/kernels.go` is already the golden reference); `n.backend` (assigned-but-never-read today) becomes live.
   - Step 3 — OpenCL backend behind its build tag validated against CPU tolerance tests.
   - Gradcheck suite is the invariant guard across all three steps.
9. **Quantization**: quantize dense layers too (existing `QuantizedDense` in `pkg/quantization/dense.go` is unused for MLPs); fix `baselineHash` to hash canonical persistence bytes instead of `json.Marshal(net)` (which serializes cell bundles as `{}`).
10. **Dataset ↔ Fit bridge**: `FitDataset(ctx context.Context, ds dataset.Dataset[T]) (uint, T, error)` (epoch = drain to `io.EOF` + `Reset`); add `Close()` to `prefetchDataset` (goroutine/fd leak); implement the documented-but-missing `ImageShaper` auto-shape via explicit `WithInputShapeFrom(ds dataset.ImageShaper)` option and fix the doc claim in `pkg/dataset/dataset.go:71`.
11. **WorkerPool**: mark deprecated (unused stub) — batch parallelism is out of scope here.

## Phase 5 — Security hardening (block E)

1. Constant-time token compare (`crypto/subtle`) in `pkg/visualization/middleware.go:18`.
2. pprof server (`pkg/nn/profiling.go`): keep `*http.Server` handles so `NN.Close` can shut them down; use a dedicated mux (not `DefaultServeMux`); Warn when binding non-loopback.
3. Decompression limits: `io.LimitReader` in `checkpoint.maybeDecompress`; IDX header sanity cap (`recLen×elemSize` bounded, overflow-checked) in `pkg/dataset/mnist.go`.
4. `http.Server` hardening for vis: `IdleTimeout`, `MaxHeaderBytes`; docs recommend `127.0.0.1` binds.

## Phase 6 — Cleanup, docs, dead code (block F)

- `applyDefaults` only substitutes on `== 0`; negative LR/LossLimit now reach `validate` and error (today `WithLearningRate(-0.5)` silently becomes 0.3).
- Delete dead code: `topologyTx`, attention `Backward` placeholder first-pass (`multihead.go:312-341`), `bufAttn`, `PositionalEncoding.lastIn`, empty `pkg/nn/api/`, `Output.CalculateValue` pre-activation-miss override (or align it), duplicate `cell.Id` for outputs (`NewOutput` writes kind into the position field).
- Doc truth pass: `Derivative` convention, `fireEvent` semantics, `Conv1D.Backward` "accumulate" claim, `StopContextCancel`, README/CHANGELOG feature matrix vs reality, MNIST example README (BatchNorm claim).
- Replace `Pause` busy-wait (`runtime.Gosched` loop in `awaitSafePoint`) with a `sync.Cond`/channel wait.
- Version bump to `v0.11.0` (`libVersion` in `pkg/nn/train.go:16`), CHANGELOG entry describing numeric breaking changes and migration notes.

## Execution order & dependencies

Phase 0 → 1 → 2 → 3 → 4 (items 1–7, then 8 SoA/backend, 9–11) → 5 → 6. Phases 5 and 6 items are independent and can interleave. Every phase ends with the full verification suite green plus `-race`. Math fixes (Phase 1–2) are done in the current cell-graph first — small reviewable diffs under the gradcheck oracle — and only then migrated to SoA storage (Phase 4.8), using the same oracle as the safety net for the migration.

## Verification (end-to-end)

1. `go test ./... -race -count=1` — all packages including `internal/verification`.
2. Gradient checks: worst relative error < 1e-4 (float64) for every topology × activation × loss combination, including conv/rnn/attention prefixes and norm/dropout paths.
3. Convergence goldens: XOR variants, multi-output, PresetMNIST on 500 samples (must now actually learn: accuracy ≳ 0.8), conv2d/rnn/attention prefix nets — loss decreasing and below thresholds.
4. Concurrency: 16-goroutine `Query` storm under `-race` — zero races, zero wrong answers; `Fit` + concurrent `Query` smoke.
5. Lifecycle: Stop→Fit resumes; AndTrain(WithLearningRate(0)) freezes weights; AddNeuron preserves surviving weights and does not panic with Adam.
6. Round-trips: nn.Save/Load query-identical (existing tolerance helpers in `pkg/compute/cpu`); checkpoint write/resume; quantized MLP output within tolerance of float net.
7. Security: vis auth timing test (statistical), gzip-bomb fixture rejected, oversized IDX header rejected.
8. Run `examples/` (xor, mnist, mnist_cnn, persistence, continuation) — build and produce sensible output; CLI `gonn train/query/verify` smoke on a CSV fixture.
