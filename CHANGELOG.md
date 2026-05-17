# Changelog

All notable changes to the GoNN library will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
release artifacts dictated by [.magic/run.md](.magic/run.md) Phase Completion / Plan Completion.

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

