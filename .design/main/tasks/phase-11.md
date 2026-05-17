---
phase: 11
name: Meta-Learning Hooks + Convolutional Layers + Dataset Formats
status: Done
subsystem: pkg/nn (meta), pkg/layer/conv (new), pkg/dataset (extend), pkg/nn (andtrain), examples/E06, examples/E10
requires:
  - phase-10 (v0.9.0 tagged; pkg/layer/norm Stable; pkg/nn callbacks Stable)
  - l1-meta-learning-hooks RFC→Stable promotion (Track A only — explicit review required via magic.spec)
  - External: MNIST IDX data files (Track C T-11C04 only — train-images-idx3-ubyte.gz + train-labels-idx1-ubyte.gz)
provides:
  - pkg/layer/conv (Conv1D[T], MaxPool1D[T], AvgPool1D[T], Flatten[T])
  - pkg/dataset (IDXReader, MNISTLoader[T])
  - pkg/nn (AndTrain, conv-prefix integration, WithConv1D/MaxPool1D/AvgPool1D/Flatten options)
  - pkg/network (AppendInputGradient method)
key_files:
  created:
    - pkg/layer/conv/conv.go
    - pkg/layer/conv/conv1d.go
    - pkg/layer/conv/pool.go
    - pkg/layer/conv/flatten.go
    - pkg/layer/conv/conv_test.go
    - pkg/dataset/mnist.go
    - pkg/nn/andtrain.go
    - pkg/nn/andtrain_test.go
    - pkg/nn/conv_test.go
    - examples/continuation/main.go
    - examples/continuation/main_test.go
    - examples/continuation/README.md
    - examples/continuation/go.mod
  modified:
    - pkg/utils/errors.go (ErrConvShapeMismatch, ErrConvPoolSizeMismatch, ErrIDXMagic, ErrMNISTRecordMismatch, ErrNetworkRunning)
    - pkg/network/propagation.go (AppendInputGradient)
    - pkg/nn/config.go (ConvPrefix field)
    - pkg/nn/nn.go (convPrefix/rawInputSize/convBuf/convGradBuf)
    - pkg/nn/options.go (WithConv1D/MaxPool1D/AvgPool1D/Flatten)
    - pkg/nn/compile.go (conv chain shape resolution + Conv1D weight init)
    - pkg/nn/train.go (runConvForward/applyConvBackward/applyConvSGD)
    - pkg/nn/query.go (pre-stage runConvForward)
    - go.work
patterns_established:
  - Conv prefix runs BEFORE Input layer; compile() resizes Input to conv outputLen; rawInputSize preserved for SetInputs
  - Conv1D weights as flattened filter-major []T (one alloc, cache-locality, sync.Pool friendly)
  - Conv backward inlined SGD (w -= lr·g); optimizer pluggability deferred to v0.11
  - MNISTLoader batches via project Dataset[T] interface (aligns with csv.go / slice.go precedent)
  - AndTrain delegates to Fit after live-cfg patch + defer restore (preserves state-machine + OnTrainEnd dispatch)
duration_minutes: ~
---

# Phase 11 — Meta-Learning Hooks + Convolutional Layers + Dataset Formats

**Status:** In Progress (Tracks B + C fully integrated; T-11C04 E06 MNIST example pending external data; gate T-11Z01 pending)
**Decomposed:** 2026-05-12
**Last Updated:** 2026-05-14 (Session 3: T-11B05 full integration done — conv prefix runs end-to-end through Train/Fit/Query; `Network.AppendInputGradient` added; 7 new integration tests green. Pending: T-11C04 E06 MNIST (needs IDX data files), T-11T02/03 final, T-11Z01 gate + v0.10.0 tag.)
**Tasks:** 10 feature + 3 validation + 1 gate = 14 total — 8 feature done, 1 validation partial.
**Specs:** l2-meta-learning-impl v0.1.0 (Draft, blocked), l2-conv-layers-impl v0.1.0 (Stable), l2-dataset-loader-impl v0.1.0 (Stable)
**Track order:** B and C are fully parallel and unblocked; A requires l1-meta-learning-hooks → Stable first;
               T-11T01 after Track A, T-11T02 after Track B, T-11T03 after Track C; Gate T-11Z01 after all.

## Track A — Meta-Learning Hooks (pkg/nn/meta.go) [Done via Phase 12]

*Goal: Implement ParamAccessor[T] + MetaLearner[T] from l2-meta-learning-impl.md.*
*Source: [l2-meta-learning-impl.md](../specifications/l2-meta-learning-impl.md)*
*Blocker: l1-meta-learning-hooks.md must be promoted RFC → Stable before this track can start.
Run `/magic.spec` to perform the RFC review and promotion.*

- [x] **T-11A01** [Completed via Phase 12 as T-12A01] — Create `pkg/nn/meta.go`:
  `ParamAccessor[T utils.Float]` interface (`Get() []T`, `Set([]T) error`, `Name() string`);
  `ScalarParam[T]` wrapper (ptr `*T`; Get returns `[]T{*ptr}`; Set validates `len==1`);
  `SliceParam[T]` wrapper (ptr `*[]T`; Get returns copy; Set validates length match);
  `MetaLearner[T]` struct with `inner *NN[T]`, `params []ParamAccessor[T]`, `FeatureFunc func(loss T, iter int) []T`.
  Add `ErrMetaLearnerShape`, `ErrMetaLearnerRunning` to `pkg/utils/errors.go`.

- [x] **T-11A02** [Completed via Phase 12 as T-12A02] — Implement `MetaLearner[T].step(loss T, iter int) error`:
  Build feature vector via `FeatureFunc` (default: `[]T{loss, T(iter)/T(maxIter)}`);
  call `inner.Query(features)` → output slice;
  validate `len(output) == len(params)` → `ErrMetaLearnerShape` if not;
  call `params[i].Set([]T{output[i]})` for each; propagate first error, continue others.

- [x] **T-11A03** [Completed via Phase 12 as T-12A03] — Wire into `pkg/nn/`:
  `pkg/nn/config.go`: add `MetaLearner *MetaLearner[T]` field.
  `pkg/nn/options.go`: add `WithMetaLearner[T](ml *MetaLearner[T]) Option[T]`.
  `pkg/nn/train.go`: after `opt.Step` call, add:
  `if nn.cfg.MetaLearner != nil { if err := nn.cfg.MetaLearner.step(loss, i); err != nil { ... } }`.
  No change to `Train()` signature or return type.

## Track B — Convolutional Layers (pkg/layer/conv/)

*Goal: Implement Conv1D[T], MaxPool1D[T], Flatten[T] from l2-conv-layers-impl.md.*
*Source: [l2-conv-layers-impl.md](../specifications/l2-conv-layers-impl.md)*

- [x] **T-11B01** — `pkg/layer/conv/conv.go` + errors. **Done 2026-05-14.**
  Local `conv.Layer[T]` interface (mirrors `pkg/layer/norm` precedent — root `pkg/layer` has no shared interface);
  `PadMode` enum (PadValid=0, PadSame=1); `outputLen` + `padSamePadding` helpers;
  compile-time assertions for Conv1D/MaxPool1D/AvgPool1D/Flatten;
  added 5 sentinel errors to `pkg/utils/errors.go`: ErrConvShapeMismatch, ErrConvPoolSizeMismatch, ErrIDXMagic, ErrMNISTRecordMismatch, ErrNetworkRunning.
  **Changes**: `pkg/layer/conv/conv.go` (+138 lines), `pkg/utils/errors.go` (+54 lines).
  **Verify**: `go build ./pkg/layer/conv/...` clean.

- [x] **T-11B02** — `pkg/layer/conv/conv1d.go`. **Done 2026-05-14.**
  Decision: **Variant A — flattened `[]T` of length `numFilters*kernelSize`**, filter-major layout (rationale: one allocation, cache-locality, sync.Pool-friendly).
  `Forward` (cross-correlation, PadValid + PadSame); `Backward` (∂L/∂W, ∂L/∂B, ∂L/∂X) accumulating into reused scratch buffers; `Init` via He-normal; `Validate` enforces CONV-1 shape; `MarshalJSON/UnmarshalJSON` for CONV-7.
  Implements `Layer[T]` interface from conv.go.
  **Changes**: `pkg/layer/conv/conv1d.go` (+325 lines).
  **Verify**: `TestConv1DGradFiniteDifference` confirms CONV-4 within 1e-4 tolerance.

- [x] **T-11B03** — `pkg/layer/conv/pool.go`. **Done 2026-05-14.**
  `MaxPool1D[T]` (argmax-tracking) + `AvgPool1D[T]` (even distribution); both non-overlapping (stride=poolSize), boundary windows truncated (no zero-padding); `Validate` enforces poolSize ≤ inLen.
  **Changes**: `pkg/layer/conv/pool.go` (+232 lines).
  **Verify**: TestMaxPool1DBackward routes gradient to argmax positions only; TestAvgPool1DBackward distributes 1/poolSize evenly.

- [x] **T-11B04** — `pkg/layer/conv/flatten.go`. **Done 2026-05-14.**
  Stateless `Flatten[T]`: Forward = identity (aliases input, no copy); Backward returns upstream unchanged on shape match, nil on mismatch.
  **Changes**: `pkg/layer/conv/flatten.go` (+50 lines).
  **Verify**: TestFlattenIdentity covers forward/backward identity and shape-mismatch path.

- [x] **T-11T02 (partial)** — Conv package validation. **Done 2026-05-14.**
  `pkg/layer/conv/conv_test.go` (+395 lines, 22 test functions). `go test ./pkg/layer/conv/...` green; **coverage 83.0%** (above 80% floor).
  Tests cover: outputLen all branches, Conv1D zero/known-kernel forward, ∂L/∂W finite-difference (CONV-4), JSON round-trip (CONV-7), Validate error paths, MaxPool/AvgPool forward/backward correctness, Flatten identity, accessor coverage, constructor clamp, shape-mismatch returns.
  **Remaining**: full network-integration tests (conv prefix in compile, training loop) deferred until T-11B05 train-loop integration lands.

- [x] **T-11B05** — Full conv-prefix integration into `pkg/nn/` + `pkg/network/`. **Done 2026-05-14.**
  Architecture: conv prefix runs as a preprocessing stage BEFORE the Input layer; `compile()` resizes the Input layer to the conv chain's `outputLen()` output, raw input size preserved on `NN[T].rawInputSize` for `SetInputs`-style validation.
  **Forward path** (`runConvForward`): raw input → chain of `cl.Forward` calls → result feeds `Network.SetInputs`.
  **Backward path**: `Network.AppendInputGradient` (new method, formula `−Σ σ'(preact_j)·miss_j·axon[i→j].weight`) → piped through conv chain in reverse → each Conv1D accumulates ∂L/∂W in its internal scratch buffers → inline SGD `w -= lr·g`. (Wiring conv weights through `optimizer.Optimizer` deferred to v0.11.)
  **Changes**:
  - `pkg/network/propagation.go` (+45 lines): `AppendInputGradient` method.
  - `pkg/nn/config.go` (+11 lines): `Config.ConvPrefix []conv.Layer[T]` field.
  - `pkg/nn/nn.go` (+18 lines): `convPrefix`, `rawInputSize`, `convBuf`, `convGradBuf` fields on `NN[T]`.
  - `pkg/nn/options.go` (+55 lines): `WithConv1D`, `WithMaxPool1D`, `WithAvgPool1D`, `WithFlatten` options.
  - `pkg/nn/compile.go` (+45 lines): conv chain shape resolution, raw input size capture, Conv1D weight init.
  - `pkg/nn/train.go` (+85 lines): `runConvForward`, `applyConvBackward`, `applyConvSGD`; `trainStep` integrates conv-FWD/BWD.
  - `pkg/nn/query.go` (+4 lines): pre-stage `runConvForward` in inference path.
  - `pkg/nn/conv_test.go` (+220 lines, 7 tests): shape resolution, error paths, training convergence, weight movement, zero-overhead pure-Dense path, MaxPool/AvgPool composability.
  **Verify**: `go build ./...` clean; `go test -count=1 ./pkg/nn/... ./pkg/network/... ./pkg/layer/conv/...` green (pre-existing flakes `TestPauseResumeCycle`, `TestRepeatBuilderBenchmark100Layer` unaffected); coverage pkg/nn **80.9%**, pkg/network **88.6%**, pkg/layer/conv **83.0%**.

## Track C — Dataset Formats + Deferred Examples

*Goal: Implement IDXReader, MNISTLoader[T], AndTrain from l2-dataset-loader-impl.md; ship E06 + E10.*
*Source: [l2-dataset-loader-impl.md](../specifications/l2-dataset-loader-impl.md)*

- [x] **T-11C01** — `pkg/dataset/mnist.go` IDXReader. **Done 2026-05-14.**
  `readIDXHeader` validates magic bytes 0-1, dtype byte set, ndim ≥ 1; supports all 6 IDX dtypes (uint8/int8/int16/int32/float/double). `IDXReader` with reusable per-record buffer, `Next()` returning `io.EOF` at end, `Reset()` returning `ErrUnsupported` (single io.Reader).
  Errors `ErrIDXMagic`, `ErrMNISTRecordMismatch`, `ErrNetworkRunning` added in earlier T-11B01 batch.
  **Changes**: `pkg/dataset/mnist.go` (+135 lines for IDX layer).
  **Verify**: TestReadIDXHeader{Valid,BadMagic,UnknownDType,ZeroNdim,ShortRead}, TestIDXReader{Iterate,ShortRecord}, TestElementSize.

- [x] **T-11C02** — `MNISTLoader[T]` in `pkg/dataset/mnist.go`. **Done 2026-05-14.**
  `MNISTLoader[T]` implements the project's `Dataset[T]` interface (batch streaming, not single-record — aligns with existing `csv.go` / `slice.go` precedent).
  Two constructors: `NewMNISTLoader` (BYO `*IDXReader` pair, no Reset support) and `NewMNISTLoaderFiles` (opens files, Reset reopens them). `Next` builds a `Batch[T]` of at most `batchSize` records; pixels normalised to [0, 1] via `T(b) / norm`. `Close` releases file handles tracked from `NewMNISTLoaderFiles`. Validation: image/label count mismatch → `ErrMNISTRecordMismatch`; nil readers / non-positive batch / non-positive norm → `ErrUserConfig`. Context cancellation honoured.
  **Changes**: `pkg/dataset/mnist.go` (+185 lines for loader layer; total file 420 lines).
  **Verify**: `go test ./pkg/dataset/...` green; **coverage 85.2%**. Tests: TestMNISTLoader{Batching,Mismatch,ArgValidation,FromFiles,FilesMissing,ContextCancelled}.

- [x] **T-11C03** — `pkg/nn/andtrain.go`. **Done 2026-05-14.**
  Decision: AndTrain signature mirrors existing `Fit` (`[]Sample[T]`, not the streaming `Dataset[T]` — aligns with how Fit consumes data, avoids streaming refactor). Snapshot cfg + opt/sched/reg pointers, apply opts to live cfg, delegate to Fit (which owns state-machine transitions + OnTrainEnd dispatch), restore on defer. Guards: `stateField == Operational` else `ErrUserConfig`; `control.Load() == Idle` else `ErrNetworkRunning`; empty samples → `ErrInputData`.
  **Changes**: `pkg/nn/andtrain.go` (+80 lines), `pkg/nn/andtrain_test.go` (+125 lines).
  **Verify**: TestAndTrain{ContinuationPreservesWeights, RestoresOriginalConfig, RejectsEmptySamples, RejectsNonOperational, RejectsRunning} — all green.

- [x] **T-11C05** — `examples/continuation/`. **Done 2026-05-14.**
  Demonstrates FMT-6 end-to-end: trains XOR via Fit, then `AndTrain` with negated targets at lower LR. Post-AndTrain predictions invert from `[0.10, 0.92, 0.91, 0.07]` → `[0.92, 0.06, 0.06, 0.95]` confirming weight continuity. `main_test.go` asserts post-AndTrain L1 distance to negated targets < distance to original.
  **Changes**: `examples/continuation/{main.go,main_test.go,README.md,go.mod}`; `go.work` updated.
  **Verify**: `go run ./examples/continuation/` produces expected output; `go test ./examples/continuation/` green.

- [x] **T-11C04** — Create `examples/mnist/` (E06). **Done 2026-05-17.**
  `main.go`: `NewMNISTLoaderFiles[float32]`; `loadSamples` with `oneHot` encoding; 784→128(ReLU+BN)→64(ReLU)→10(Sigmoid);
  `Fit(1 epoch)` + `AndTrain(lr=0.001)`; `argmax` predicted digit. `-images`/`-labels`/`-n` flags.
  `README.md`: IDX binary format, one-hot encoding rationale, BatchNorm training/eval distinction, AndTrain fine-tuning pattern, MNIST download instructions.
  `go.mod` + `go.work` updated.
  **Changes**: `examples/mnist/main.go` (+111 lines), `examples/mnist/README.md` (+90 lines), `examples/mnist/go.mod` (+6 lines), `go.work` (+1 line).
  **Verify**: `go build ./examples/mnist/...` → exit 0 (smoke-run deferred; IDX data not committed).

## Validation Tasks

- [x] **T-11T01** [Completed via Phase 12 as T-12T01] — Meta-Learning validation (after Track A):
  - `go test -count=1 -race ./pkg/nn/...` — all tests green including new meta tests.
  - Verify `MetaLearner` with a 2→1 inner NN tuning outer learning rate converges faster than static LR on a synthetic task.
  - Verify `ErrMetaLearnerShape` fires when inner output size ≠ registered params.
  - Verify `ErrMetaLearnerRunning` fires when `WithMetaLearner` called during training.

- [x] **T-11T02** — Convolutional layers validation. **Done 2026-05-17.**
  - `go test -count=1 ./pkg/layer/conv/...` → exit 0, 22 tests pass (race flag skipped: gcc not in PATH on Windows; CI must re-run with -race).
  - Coverage `pkg/layer/conv/` = **83.3%** ✓ (≥80 % floor).
  - pkg/nn 81.8%, pkg/network 88.5% (conv integration tests included).
  - WithConv1D → outputLen formula verified by TestOutputLen; CONV-1 10→8 in TestConv1DKnownKernel ✓.
  - JSON round-trip: TestConv1DJSONRoundTrip ✓ (CONV-7).
  - Gradient finite-difference: TestConv1DGradFiniteDifference ≤1e-4 ✓ (CONV-4).
  - `go build ./...` → exit 0 (including examples/mnist) ✓.
  - Note: TestXORTwoHiddenConvergence (stochastic convergence) occasionally flakes on low-entropy seed — pre-existing per STATE.md; passes 5/6 runs.

- [x] **T-11T03** — Dataset loader + examples validation. **Done 2026-05-17** (E06 smoke deferred).
  - `go test -count=1 ./pkg/dataset/...` → exit 0, 16 tests pass (gcc not in PATH; -race in CI).
  - Coverage `pkg/dataset/` = **85.2%** ✓ (≥80 % floor).
  - ErrIDXMagic: TestReadIDXHeaderBadMagic ✓.
  - ErrMNISTRecordMismatch: TestMNISTLoaderMismatch ✓.
  - `go run ./examples/E06_mnist/` → **deferred**: IDX data not committed; code builds clean. See examples/mnist/README.md for download instructions.
  - `go test ./examples/continuation/...` → TestAndTrainContinuation PASS ✓.

## Gate

- [x] **T-11Z01** — Phase 11 gate. **Done 2026-05-17.**
  - `go build ./...` → exit 0 ✓ (zero regressions across all 19 packages).
  - `go test -count=1 ./...` → all 19 packages green ✓ (-race deferred: gcc not in PATH on Windows; CI must re-run with -race).
  - Coverage floor met: `pkg/layer/conv/` **83.3%** ✓, `pkg/dataset/` **85.2%** ✓, `pkg/nn/` **81.5%** ✓.
  - Track A (l1-meta-learning-hooks still RFC): T-11A01..T-11A03 + T-11T01 marked `[Deferred to Phase 12]`.
  - `CHANGELOG.md` v0.10.0 entry written: conv layers, dataset loader, AndTrain, E06 MNIST example.
  - `v0.10.0` git tag: pending user's `git tag -a v0.10.0 -m "..."` per l1-release-policy §5.4.
