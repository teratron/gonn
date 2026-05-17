# Project State

<!-- STATE.md — live project memory. Read FIRST in every workflow session. -->
<!-- Maximum 100 lines. Agent updates AFTER each completed action. -->

**Workspace:** main
**Project Version:** 0.9.0 released; 0.10.0 in progress (Phase 11 Tracks B+C nearly done)
**Updated:** 2026-05-17 07:52
**Phase:** 11 — Meta-Learning Hooks + Conv Layers + Dataset Formats
**Status:** Active

## Current Position

- **Task:** T-11C04 (E06 MNIST example — code+README scaffold ready to write; smoke-run gated on user-supplied IDX data)
- **Spec:** Track B (l2-conv-layers-impl v0.1.0) integrated end-to-end through compile/train/query. Track C (l2-dataset-loader-impl v0.1.0) IDXReader + MNISTLoader[T] + AndTrain live; examples/continuation/ smoke-clean. Track A (l2-meta-learning-impl v0.1.0) Draft — blocked on l1-meta-learning-hooks RFC→Stable.
- **Next Action:** Run /magic-run main to close Phase 11: (a) T-11C04 (E06 MNIST example — needs user-supplied IDX data, but code+README can land first), (b) T-11T02/T-11T03 final validation, (c) gate T-11Z01 (build + race + ≥80% cover) + CHANGELOG v0.10.0 + v0.10.0 tag. Track A (Meta-Learning) remains blocked — promote l1-meta-learning-hooks RFC→Stable via /magic-spec when ready.

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
Phase 11 (Active):  [8/14]    █████░░░  57%   (Track A 0/3 blocked; Track B 5/5 done; Track C 3/4 done + 1 pending; T-11T02 partial; T-11T03 + gate pending)
Overall:            [181/187] ███████░  97%
```

## Recent Decisions

- 2026-05-10 **Decision:** Phase 8 complete. Track A: `pkg/optimizer/exponential_lr.go` — `ExponentialLR[T]` (lr₀ × gamma^t), LRS-1..LRS-6, default PerEpoch, JSON SaveState/LoadState; `pkg/optimizer/` coverage 88.2%. Track B: `cmd/gonn/` — `main.go` (dispatch), `train.go`, `query.go`, `verify.go`, `version.go`, `exitcode.go`, `load.go`, `csv.go` — full CLI binary with train/query/verify/version subcommands, `--precision float32|float64` generic dispatch, 64 MB streaming threshold, `--json` output, exit-code contract (0-7), XOR end-to-end smoke test; coverage 83.7%. l2-lr-scheduling-impl.md bumped to v1.1.0. CHANGELOG.md v0.7.0 entry written. Gate T-8Z01: `go build ./...` clean; `go test ./...` all 16 packages green; all packages ≥80%.
- 2026-05-10 **Decision:** Phase 7 complete. Track A: `pkg/optimizer/` extended with `Scheduler[T]` interface, `BindScheduler[T]`, `LearningRateSetter[T]` optional extension, and four scheduler types — `StepLR[T]` (step decay), `WarmUpLR[T]` (linear ramp, PerStep default), `CosineAnnealingLR[T]` (cosine decay), `ChainScheduler[T]` (sequential composition). All four existing optimizers (SGD/Adam/RMSProp/SGDMomentum) implement `LearningRateSetter[T]` via `SetLearningRate(T)`. `pkg/optimizer/scheduler_test.go` covers all scheduler types, BindScheduler wiring, Granularity defaults, and SaveState/LoadState round-trips; coverage 88.9 %. Track B: `pkg/nn/builder.go` + `pkg/nn/options.go` extended with bulk constructors `Repeat`/`Pattern`/`HiddenLayers` (Builder, setter/append semantics distinguished) and `Repeat[T]`/`Pattern[T]`/`WithHiddenLayers[T]` (Options, append); `WithScheduler` added to both; `pkg/nn/train.go` dispatches `sched.Step()` per `Granularity()` (PerEpoch after epoch, PerStep per batch); `pkg/nn/config.go` adds `Scheduler` field; `pkg/nn/phase7_test.go` covers 11 test functions + 2 benchmarks; coverage 86.4 %. Track C: `skills/gonn/` created with `SKILL.md` (10 sections, YAML frontmatter), 3 example files (builder-xor, options-mnist, deep-network), and 2 resource files (api-reference, conventions). Gate T-7Z01: `go build ./...` clean; `go test ./pkg/...` all green; all packages ≥80 %; skills directory matches spec §5.1; 3 orphaned specs resolved.
- 2026-05-08 **Decision:** Phase 6 complete. Tracks A+B (optimizer + regularizer + WeightInit fix): `pkg/optimizer/` (SGD/Adam/RMSProp/SGDMomentum, 98.9% cover, 0 allocs/op benchmarks), `pkg/regularizer/` (L1/L2/Dropout/Compose, 80.0% cover), `pkg/nn` wired via `WithOptimizer`/`WithRegularizer`/`trainStep()`. Track C: CHANGELOG.md v0.6.0, README.md updated with full v0.6 API docs. Validation: TestWeightInitRanges, TestRegularizerConvergence, TestInferenceNoDropout, TestOptimizerIntegration all green. Gate T-6Z01: all packages ≥80% cover (`network` 96.3%, `nn` 86.4%). Tag v0.6.0 created. Pre-existing TestPauseResumeCycle timing flake documented in CHANGELOG Known Issues.

## Recent Decisions

- 2026-05-12 **Decision:** Phase 10 complete. Track A: `pkg/layer/norm/` — `Normalizer[T]` interface, `BatchNorm[T]` (EMA stats, affine, JSON), `LayerNorm[T]` (per-sample, stateless), `GroupNorm[T]` (G-group, divisibility guard), shared helpers `stddev`/`applyAffine`; coverage 82%. Track B: `pkg/nn/callbacks.go` — `ErrStopTraining`, `StopReason` (6 values), `CallbackContext[T]`, `CallbackFn[T]`, `CallbackRegistry[T]`, `invokeOne` (panic recovery CB-5), `fireEvent` (nil short-circuit CB-3, ordered CB-7), `fireOnTrainEnd` (defer CB-8). `pkg/nn/train.go` Fit wired with deferred OnTrainEnd, per-epoch OnImprovementFound/OnIterationEnd, ErrStopTraining → rollback. `pkg/nn/options.go` + `config.go` + `nn.go` + `compile.go` wired for both norm and callbacks. `pkg/utils/errors.go` adds `ErrCallbackPanic`. `pkg/nn/callbacks_test.go` 15 tests covering all CB invariants; nn coverage 80%. CHANGELOG.md v0.9.0 written. Gate T-10Z01: `go build ./...` clean; all Phase 10 packages ≥80%.
- 2026-05-10 **Decision:** Phase 9 complete. Track A: `pkg/optimizer/metric_scheduler.go` (MetricScheduler[T] interface + boundScheduler forwarding), `reduce_on_plateau.go` (patience/factor/threshold/minLR/mode, PerEpoch), `one_cycle_lr.go` (3-phase warm-up/cosine/hold, PerStep); optimizer coverage 90.3%. Track B: `pkg/network/topology.go` (TopologyMode, topologyTx rollback, AddNeuron/RemoveNeuron/AddHiddenLayer/RemoveHiddenLayer + successor rebalancing); 6 DYN error sentinels in utils/errors.go; DYN-2 wrappers in pkg/nn/topology.go; network coverage 92.9%. Track C: pkg/utils/logger.go rewritten to GoLogger slog adapter (LevelTrace=-8, nil-safe discard fallback); pkg/visualization/ new package with 6 endpoints + auth/CORS middleware; nn wired via WithLogger/WithVisualizationEndpoint/WithVisualizationToken/WithVisualizationCORS; Close() stops vis server; visualization coverage 90.0%; nn coverage 85.9%. CHANGELOG.md v0.8.0 written. l2-lr-scheduling-impl.md bumped to v1.2.0.

## Blockers

- **Track A (Meta-Learning)**: l1-meta-learning-hooks still RFC v0.3.0 — RFC→Stable review required via /magic-spec before T-11A01..T-11A03 can start (or mark them `[Deferred to Phase 12]` at gate per T-11Z01).
- **T-11C04 (E06 MNIST)**: external — needs user-supplied `train-images-idx3-ubyte.gz` + `train-labels-idx1-ubyte.gz`. Code template + README can land first; smoke-run deferred.

## Blocking Constraints

- Note: race detector requires CGO on Windows (gcc not in PATH); tests run without -race locally — gate T-11Z01 must run -race in CI.
- Note: TestPauseResumeCycle + TestMultiHiddenXOR + TestRepeatBuilderBenchmark100Layer are pre-existing timing-flaky tests, unaffected by Phase 11 work.
- **Engine drift**: `.magic/.version` = 2.1.27 vs INDEX.md snapshot 2.1.25 — acknowledged but snapshot held stale per §1 n-branch. Run /magic-analyze when ready to revalidate.

## Session Continuity

**Last Session Ended:** 2026-05-17 (sync via /magic-task — no execution)
**Handoff File:** none
**Bootstrap Mode:** false (Phase 11 mid-flight; next = /magic-run main for T-11C04 + T-11T03 + gate)
