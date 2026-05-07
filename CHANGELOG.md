# Changelog

All notable changes to the GoNN library will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
release artifacts dictated by [.magic/run.md](.magic/run.md) Phase Completion / Plan Completion.

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
