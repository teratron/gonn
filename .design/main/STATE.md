# Project State

<!-- STATE.md — live project memory. Read FIRST in every workflow session. -->
<!-- Maximum 100 lines. Agent updates AFTER each completed action. -->

**Workspace:** main
**Project Version:** 0.11.0 released
**Updated:** 2026-05-18 08:40
**Phase:** 15 — Recurrent Foundation + GPU Skeleton + AI-Meta Linter
**Status:** Todo (Active)

## Current Position

- **Task:** Phase 15 scoped — 14 atomic tasks (T-15A01..A04 / B01..B03 / C01..C03 / T01..T03 / Z01) ready for execution
- **Spec:** l1-recurrent-layers Stable v0.1.0; l2-recurrent-impl Stable v0.1.0; l2-backend-gpu Stable v0.1.0; l2-aimeta-linter Stable v0.1.0; l2-ai-doc-metadata Stable v1.0.0 (promoted from RFC)
- **Next Action:** Run /magic-run to execute Phase 15 — start with T-15A01 (utils.Orthogonal) + T-15B01 (gpu umbrella) + T-15C01 (pkg/aimeta grammar) in parallel

## Progress

```
Phase 1  (Done):    [22/22]   ████████ 100%
Phase 2  (Done):    [26/26]   ████████ 100%
Phase 3  (Done):    [19/19]   ████████ 100%
Phase 4  (Done):    [14/14]   ████████ 100%
Phase 5  (Done):    [23/23]   ████████ 100%   (all tracks + gate complete)
Phase 6  (Done):    [19/19]   ████████ 100%   (all tracks + validation + gate + tag)
Phase 7  (Done):    [16/16]   ████████ 100%   (Tracks A+B+C + validation + gate)
Phase 8  (Done):    [9/9]     ████████ 100%   (Track A: ExponentialLR + Track B: CLI binary + gate)
Phase 9  (Done):    [15/15]   ████████ 100%   (Tracks A+B+C + validation + gate)
Phase 10 (Done):    [10/10]   ████████ 100%   (Tracks A+B + validation + gate; norm + callbacks)
Phase 11 (Done):    [14/14]   ████████ 100%   (Track A 3/3 completed via Phase 12; Track B 5/5; Track C 5/5; gate done)
Phase 12 (Done):    [5/5]     ████████ 100%   (Track A 3/3 done; T-12T01 + T-12Z01 done)
Phase 13 (Done):    [2/2]     ████████ 100%   (L1 + L2 spec authoring; gate done)
Phase 14 (Done):    [12/12]   ████████ 100%   (Tracks A+B+C+D + T01 + Z01; v0.12.0 RC)
Phase 15 (Todo):    [0/14]    ░░░░░░░░   0%   (Tracks A+B+C foundation; v0.13.0 target)
Overall:            [206/220] ███████░  93.6%
```

## Recent Decisions

- 2026-05-18 **Decision:** Phase 15 scoped via /magic-task. Three fully parallel foundation tracks: A (Recurrent — SimpleRNN+LSTM with BPTT and `utils.Orthogonal`), B (GPU — `pkg/compute/gpu/` umbrella + opencl/ skeleton + Dense Forward kernel), C (Linter — `pkg/aimeta` + `cmd/lint-aimeta` + first hook in `pkg/utils`). 14 atomic tasks. Per @role:planner audit: full spec implementation plans (Recurrent α-θ, GPU A-G, Linter A-D) compressed to foundation scope; GRU/LastStep/CUDA/perf-gate/resolver/rollout-2-5 deferred to Phase 16+. All 3 new specs authored Stable in same session via prior /magic-spec; l2-ai-doc-metadata promoted RFC v0.1.0 → Stable v1.0.0. PLAN v2.12.0 → v2.13.0; TASKS v2.10.1 → v2.11.0; INDEX v2.13.1 → v2.14.0 (61 specs).
- 2026-05-17 **Decision:** Phase 14 complete. Track A: `conv2d.go` (Conv2D[T] Forward/Backward/Init/JSON, CONV2D-C9 CHW layout, PERF-4 zero-alloc Backward), `pool2d.go` (MaxPool2D argmax routing + AvgPool2D even distribution), `flatten2d.go` (stateless identity reshape). Track B: `WithConv2D/MaxPool2D/AvgPool2D/Flatten2D/InputShape` options + `setupConv2DShapes` CHW pre-pass in compile(), isqrt auto-inference (MNIST 784→1×28×28). Track C: `ImageShaper` interface in dataset package + `WithImageShape`/`ImageShape()` on MNISTLoader. Track D: `examples/mnist_cnn/` E16 (LeNet-style, two-epoch Fit+AndTrain). T-14T01: 4-case FD gradient check all PASS. Gate T-14Z01: `go build ./...` clean; all 18+1 packages green; `pkg/layer/conv/` 81.9%; `pkg/dataset/` 85.9%; `pkg/nn/` 77.1% (pre-existing). CHANGELOG v0.12.0 written.
- 2026-05-17 **Decision:** Phase 13 complete. Provides: l1-conv-2d-layers Stable v0.2.0 (9 invariants CONV2D-1..9, CHW layout) + l2-conv-2d-impl Stable v0.1.0 (filter-major flat []T storage, all 9 invariants mapped, MNIST adapter requirement in §6). Spec-critic: zero Substantive Compliance failures. check-prerequisites ok:true. Implementation tracks deferred to Phase 14.

- 2026-05-10 **Decision:** Phase 8 complete. Track A: `pkg/optimizer/exponential_lr.go` — `ExponentialLR[T]` (lr₀ × gamma^t), LRS-1..LRS-6, default PerEpoch, JSON SaveState/LoadState; `pkg/optimizer/` coverage 88.2%. Track B: `cmd/gonn/` — `main.go` (dispatch), `train.go`, `query.go`, `verify.go`, `version.go`, `exitcode.go`, `load.go`, `csv.go` — full CLI binary with train/query/verify/version subcommands, `--precision float32|float64` generic dispatch, 64 MB streaming threshold, `--json` output, exit-code contract (0-7), XOR end-to-end smoke test; coverage 83.7%. l2-lr-scheduling-impl.md bumped to v1.1.0. CHANGELOG.md v0.7.0 entry written. Gate T-8Z01: `go build ./...` clean; `go test ./...` all 16 packages green; all packages ≥80%.
- 2026-05-10 **Decision:** Phase 7 complete. Track A: `pkg/optimizer/` extended with `Scheduler[T]` interface, `BindScheduler[T]`, `LearningRateSetter[T]` optional extension, and four scheduler types — `StepLR[T]` (step decay), `WarmUpLR[T]` (linear ramp, PerStep default), `CosineAnnealingLR[T]` (cosine decay), `ChainScheduler[T]` (sequential composition). All four existing optimizers (SGD/Adam/RMSProp/SGDMomentum) implement `LearningRateSetter[T]` via `SetLearningRate(T)`. `pkg/optimizer/scheduler_test.go` covers all scheduler types, BindScheduler wiring, Granularity defaults, and SaveState/LoadState round-trips; coverage 88.9 %. Track B: `pkg/nn/builder.go` + `pkg/nn/options.go` extended with bulk constructors `Repeat`/`Pattern`/`HiddenLayers` (Builder, setter/append semantics distinguished) and `Repeat[T]`/`Pattern[T]`/`WithHiddenLayers[T]` (Options, append); `WithScheduler` added to both; `pkg/nn/train.go` dispatches `sched.Step()` per `Granularity()` (PerEpoch after epoch, PerStep per batch); `pkg/nn/config.go` adds `Scheduler` field; `pkg/nn/phase7_test.go` covers 11 test functions + 2 benchmarks; coverage 86.4 %. Track C: `skills/gonn/` created with `SKILL.md` (10 sections, YAML frontmatter), 3 example files (builder-xor, options-mnist, deep-network), and 2 resource files (api-reference, conventions). Gate T-7Z01: `go build ./...` clean; `go test ./pkg/...` all green; all packages ≥80 %; skills directory matches spec §5.1; 3 orphaned specs resolved.
- 2026-05-08 **Decision:** Phase 6 complete. Tracks A+B (optimizer + regularizer + WeightInit fix): `pkg/optimizer/` (SGD/Adam/RMSProp/SGDMomentum, 98.9% cover, 0 allocs/op benchmarks), `pkg/regularizer/` (L1/L2/Dropout/Compose, 80.0% cover), `pkg/nn` wired via `WithOptimizer`/`WithRegularizer`/`trainStep()`. Track C: CHANGELOG.md v0.6.0, README.md updated with full v0.6 API docs. Validation: TestWeightInitRanges, TestRegularizerConvergence, TestInferenceNoDropout, TestOptimizerIntegration all green. Gate T-6Z01: all packages ≥80% cover (`network` 96.3%, `nn` 86.4%). Tag v0.6.0 created. Pre-existing TestPauseResumeCycle timing flake documented in CHANGELOG Known Issues.

## Recent Decisions

- 2026-05-12 **Decision:** Phase 10 complete. Track A: `pkg/layer/norm/` — `Normalizer[T]` interface, `BatchNorm[T]` (EMA stats, affine, JSON), `LayerNorm[T]` (per-sample, stateless), `GroupNorm[T]` (G-group, divisibility guard), shared helpers `stddev`/`applyAffine`; coverage 82%. Track B: `pkg/nn/callbacks.go` — `ErrStopTraining`, `StopReason` (6 values), `CallbackContext[T]`, `CallbackFn[T]`, `CallbackRegistry[T]`, `invokeOne` (panic recovery CB-5), `fireEvent` (nil short-circuit CB-3, ordered CB-7), `fireOnTrainEnd` (defer CB-8). `pkg/nn/train.go` Fit wired with deferred OnTrainEnd, per-epoch OnImprovementFound/OnIterationEnd, ErrStopTraining → rollback. `pkg/nn/options.go` + `config.go` + `nn.go` + `compile.go` wired for both norm and callbacks. `pkg/utils/errors.go` adds `ErrCallbackPanic`. `pkg/nn/callbacks_test.go` 15 tests covering all CB invariants; nn coverage 80%. CHANGELOG.md v0.9.0 written. Gate T-10Z01: `go build ./...` clean; all Phase 10 packages ≥80%.
- 2026-05-10 **Decision:** Phase 9 complete. Track A: `pkg/optimizer/metric_scheduler.go` (MetricScheduler[T] interface + boundScheduler forwarding), `reduce_on_plateau.go` (patience/factor/threshold/minLR/mode, PerEpoch), `one_cycle_lr.go` (3-phase warm-up/cosine/hold, PerStep); optimizer coverage 90.3%. Track B: `pkg/network/topology.go` (TopologyMode, topologyTx rollback, AddNeuron/RemoveNeuron/AddHiddenLayer/RemoveHiddenLayer + successor rebalancing); 6 DYN error sentinels in utils/errors.go; DYN-2 wrappers in pkg/nn/topology.go; network coverage 92.9%. Track C: pkg/utils/logger.go rewritten to GoLogger slog adapter (LevelTrace=-8, nil-safe discard fallback); pkg/visualization/ new package with 6 endpoints + auth/CORS middleware; nn wired via WithLogger/WithVisualizationEndpoint/WithVisualizationToken/WithVisualizationCORS; Close() stops vis server; visualization coverage 90.0%; nn coverage 85.9%. CHANGELOG.md v0.8.0 written. l2-lr-scheduling-impl.md bumped to v1.2.0.

## Blockers

- **E06 smoke-run**: `go run ./examples/mnist/` deferred — needs user-supplied IDX data (see examples/mnist/README.md for download instructions). Not a Phase 12 blocker.

## Blocking Constraints

- Note: race detector requires CGO on Windows (gcc not in PATH); tests run without -race locally — gate T-11Z01 must run -race in CI.
- Note: TestPauseResumeCycle + TestMultiHiddenXOR + TestRepeatBuilderBenchmark100Layer are pre-existing timing-flaky tests, unaffected by Phase 11 work.
- Engine drift resolved 2026-05-18: INDEX.md snapshot 2.1.25 → 2.1.27 via /magic-analyze.

## Recent Decisions

- 2026-05-17 **Decision:** Phase 12 complete. Track A: `pkg/nn/meta.go` — `ParamAccessor[T]`, `ScalarParam[T]`, `SliceParam[T]`, `FeatureFunc[T]`, `DefaultFeatureFunc[T]`, `MetaLearner[T]` with `step()` (continue-on-error); `pkg/utils/errors.go` — `ErrMetaLearnerShape`, `ErrMetaLearnerRunning`; `pkg/nn/config.go` + `options.go` — `MetaLearner` field + `WithMetaLearner` option + `ErrMetaLearnerRunning` compile guard; `pkg/nn/train.go` — advisory hook in Fit after opt.Step before OnIterationEnd; `pkg/nn/meta_test.go` — 14 tests; `pkg/nn` coverage 82.2 %. CHANGELOG.md v0.11.0 written. Gate T-12Z01: `go build ./...` clean; all 19 packages green; `pkg/nn` ≥ 80 %.

## Session Continuity

**Last Session Ended:** 2026-05-17 (Phase 14 complete — v0.12.0 gate passed)
**Handoff File:** none
**Bootstrap Mode:** false (Phase 14 Done; next = user runs `git tag -a v0.12.0`, then /magic-task main to scope Phase 15)
