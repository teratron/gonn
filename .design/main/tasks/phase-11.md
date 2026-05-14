---
phase: 11
name: Meta-Learning Hooks + Convolutional Layers + Dataset Formats
status: Todo
subsystem: pkg/nn (meta), pkg/layer/conv (new), pkg/dataset (extend), pkg/nn (andtrain), examples/E06, examples/E10
requires:
  - phase-10 (v0.9.0 tagged; pkg/layer/norm Stable; pkg/nn callbacks Stable)
  - l1-meta-learning-hooks RFC→Stable promotion (Track A only — explicit review required via magic.spec)
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 11 — Meta-Learning Hooks + Convolutional Layers + Dataset Formats

**Status:** In Progress (Track B core landed; Track B integration + Track C pending)
**Decomposed:** 2026-05-12
**Last Updated:** 2026-05-14 (Track B T-11B01..B04 implemented and tested; T-11T02 partial — package-level conv tests 83% coverage. T-11B05 train-loop integration deferred; Track C pending.)
**Tasks:** 10 feature + 3 validation + 1 gate = 14 total — 4 feature done, 1 validation partial.
**Specs:** l2-meta-learning-impl v0.1.0 (Draft, blocked), l2-conv-layers-impl v0.1.0 (Stable), l2-dataset-loader-impl v0.1.0 (Stable)
**Track order:** B and C are fully parallel and unblocked; A requires l1-meta-learning-hooks → Stable first;
               T-11T01 after Track A, T-11T02 after Track B, T-11T03 after Track C; Gate T-11Z01 after all.

## Track A — Meta-Learning Hooks (pkg/nn/meta.go) [BLOCKED]

*Goal: Implement ParamAccessor[T] + MetaLearner[T] from l2-meta-learning-impl.md.*
*Source: [l2-meta-learning-impl.md](../specifications/l2-meta-learning-impl.md)*
*Blocker: l1-meta-learning-hooks.md must be promoted RFC → Stable before this track can start.
Run `/magic.spec` to perform the RFC review and promotion.*

- [ ] **T-11A01** [BLOCKED] — Create `pkg/nn/meta.go`:
  `ParamAccessor[T utils.Float]` interface (`Get() []T`, `Set([]T) error`, `Name() string`);
  `ScalarParam[T]` wrapper (ptr `*T`; Get returns `[]T{*ptr}`; Set validates `len==1`);
  `SliceParam[T]` wrapper (ptr `*[]T`; Get returns copy; Set validates length match);
  `MetaLearner[T]` struct with `inner *NN[T]`, `params []ParamAccessor[T]`, `FeatureFunc func(loss T, iter int) []T`.
  Add `ErrMetaLearnerShape`, `ErrMetaLearnerRunning` to `pkg/utils/errors.go`.

- [ ] **T-11A02** [BLOCKED] — Implement `MetaLearner[T].step(loss T, iter int) error`:
  Build feature vector via `FeatureFunc` (default: `[]T{loss, T(iter)/T(maxIter)}`);
  call `inner.Query(features)` → output slice;
  validate `len(output) == len(params)` → `ErrMetaLearnerShape` if not;
  call `params[i].Set([]T{output[i]})` for each; propagate first error, continue others.

- [ ] **T-11A03** [BLOCKED] — Wire into `pkg/nn/`:
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

- [ ] **T-11B05** — Wire into `pkg/nn/`:
  `pkg/nn/config.go`: add `ConvPrefix []layer.Layer[T]` field.
  `pkg/nn/options.go`: add `WithConv1D[T]`, `WithMaxPool1D[T]`, `WithFlatten[T]` options
  (each appends a new instance to `cfg.ConvPrefix`).
  `pkg/nn/compile.go`: prepend `cfg.ConvPrefix` before the Dense hidden stack;
  call `outputLen()` on each conv layer to compute the input size for the first Dense layer.

## Track C — Dataset Formats + Deferred Examples

*Goal: Implement IDXReader, MNISTLoader[T], AndTrain from l2-dataset-loader-impl.md; ship E06 + E10.*
*Source: [l2-dataset-loader-impl.md](../specifications/l2-dataset-loader-impl.md)*

- [ ] **T-11C01** — Create `pkg/dataset/mnist.go`:
  `readIDXHeader(r io.Reader) (idxHeader, error)` — reads 4-byte magic + dimension bytes;
  validate bytes 0-1 == 0x00, dtype in `{0x08,0x09,0x0B,0x0C,0x0D,0x0E}` → `ErrIDXMagic` otherwise;
  read `ndim` big-endian int32 dimensions; compute `recLen = product(dims[1:])`.
  `IDXReader` struct: wraps `io.Reader` + `idxHeader`; `Next() ([]byte, error)` reads one record.
  Add `ErrIDXMagic`, `ErrMNISTRecordMismatch`, `ErrNetworkRunning` to `pkg/utils/errors.go`.
  Begin `pkg/dataset/mnist_test.go`: valid IDX header parse, invalid magic error, record count.

- [ ] **T-11C02** — Create `MNISTLoader[T]` in `pkg/dataset/mnist.go`:
  `type MNISTLoader[T utils.Float] struct { images, labels *IDXReader; norm T }`;
  `NewMNISTLoader[T](imageFile, labelFile string, norm T) (*MNISTLoader[T], error)`:
  open both files, read headers, validate counts match (FMT-5);
  `Next() (input []T, target []T, err error)`: read one image record, normalise pixels by `norm`,
  read one label record, cast label byte to `T`; return `io.EOF` when exhausted.
  Implement `dataset.DataSet[T]` interface (streaming, bounded memory).
  Extend test: full record iteration, pixel normalisation, io.EOF on completion.

- [ ] **T-11C03** — Create `pkg/nn/andtrain.go`:
  `func (nn *NN[T]) AndTrain(ds dataset.DataSet[T], opts ...Option[T]) (Result[T], error)`:
  check `nn.ctrl.State() == Idle` → `ErrNetworkRunning` if not;
  apply `opts` to a shallow copy of `nn.cfg` (not mutating base config);
  call shared inner `train(cfg, ds)` helper (refactor `Train` to extract this);
  callbacks remain active; `OnTrainEnd` fires at completion; state returns to `Idle`.
  Add `andtrain_test.go`: continuation preserves weights, fresh convergence counters,
  `OnTrainEnd` fires for each call, error on Running state.

- [ ] **T-11C04** — Create `examples/E06_mnist/`:
  `main.go`: load MNIST train set via `MNISTLoader`, build 784→128→64→10 network with `WithBatchNorm`,
  train with `AndTrain` for a second epoch, query a sample image, print predicted digit.
  `README.md`: explains MNIST IDX format, one-hot encoding, why BatchNorm helps deep nets.

- [ ] **T-11C05** — Create `examples/E10_continuation/`:
  `main.go`: pre-train XOR network for 500 iterations, call `AndTrain` with a second dataset
  (negated XOR), demonstrate weight preservation and continued convergence.
  `README.md`: explains `AndTrain` semantics, when to use continuation vs rebuilding.

## Validation Tasks

- [ ] **T-11T01** — Meta-Learning validation (after Track A):
  - `go test -count=1 -race ./pkg/nn/...` — all tests green including new meta tests.
  - Verify `MetaLearner` with a 2→1 inner NN tuning outer learning rate converges faster than static LR on a synthetic task.
  - Verify `ErrMetaLearnerShape` fires when inner output size ≠ registered params.
  - Verify `ErrMetaLearnerRunning` fires when `WithMetaLearner` called during training.

- [ ] **T-11T02** — Convolutional layers validation (after Track B):
  - `go test -count=1 -race ./pkg/layer/conv/...` — all tests green.
  - Coverage `pkg/layer/conv/` ≥80 %.
  - Verify `WithConv1D(3, 3, 1, PadValid)` on input length 10 produces output length 8 (CONV-1).
  - Verify JSON round-trip: compiled network with conv layers serialises and reloads correctly.
  - Verify gradient finite-difference check passes for `Conv1D.CalculateError` (CONV-4).
  - `go build ./...` — zero regressions.

- [ ] **T-11T03** — Dataset loader + examples validation (after Track C):
  - `go test -count=1 -race ./pkg/dataset/...` — all tests green.
  - Coverage `pkg/dataset/` ≥80 %.
  - Verify `ErrIDXMagic` on tampered magic bytes.
  - Verify `ErrMNISTRecordMismatch` when image/label counts differ.
  - `go run ./examples/E06_mnist/` — executes without panic (smoke test; no accuracy threshold).
  - `go run ./examples/E10_continuation/` — executes without panic.

## Gate

- [ ] **T-11Z01** — Phase 11 gate:
  - `go build ./...` clean (zero regressions across all packages).
  - `go test -count=1 -race ./...` all green.
  - Every new/modified package ≥80 % line coverage:
    `pkg/layer/conv/` (new), `pkg/dataset/` (mnist addition), `pkg/nn/` (meta + andtrain additions).
  - Track A tasks: skipped if l1-meta-learning-hooks is still RFC at gate time — mark `[Deferred to Phase 12]`.
  - `CHANGELOG.md` v0.10.0 entry written (conv layers + dataset loader + AndTrain; meta-learning if unblocked).
  - `v0.10.0` git tag created per `l1-release-policy.md §5.4`.
