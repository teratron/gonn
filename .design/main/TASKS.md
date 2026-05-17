# Master Task Index (Registry)

**Version:** 2.8.0
**Project Version:** 0.11.0 released
**Generated:** 2026-04-29
**Last Updated:** 2026-05-17
**Based on:** .design/main/PLAN.md v2.8.0
**Based on RULES:** .design/RULES.md v1.3.0
**Execution Mode:** Parallel (C3)
**Status:** Active

## Overview

Tactical registry of all phases and their statuses. Atomic checklists (`T-XXXX`) live in per-phase files inside `tasks/`. Progress (`[x]`, `[/]`, `[!]`) is recorded in those files first; this index summarizes phase-level status only.

## Active Phases

| Phase | Description | Status |
| :--- | :--- | :--- |
| [Phase 1](tasks/phase-1.md) | Foundation Rewrite (Track A) — `pkg/utils/{errors,init}`, `pkg/neuron`, `pkg/layer`, `pkg/network` | `Done` (2026-04-29) |
| [Phase 2](tasks/phase-2.md) | Public Facade Restoration (Track B) — `pkg/nn` | `Done` (2026-04-30) |
| [Phase 3](archives/tasks/phase-3.md) | New Capability Packages (Track C) — persistence, checkpoint, dataset, compute, perf | `Done (Archived)` (2026-05-01) |
| [Phase 4](tasks/phase-4.md) | Examples Catalog (Track D) — `examples/E01..E15` (v0.5 scope: 7 entries) | `Done` (2026-05-02) |
| [Phase 5](tasks/phase-5.md) | Multi-Hidden Topology (v0.6) — `pkg/network` chain + compile() lift + weights 1.1.0 + 6 catalog examples | `Done` (2026-05-06) |
| [Phase 6](tasks/phase-6.md) | Feature Expansion + v0.6 Release — `pkg/optimizer/`, `pkg/regularizer/`, axon WeightInit fix, CHANGELOG, v0.6.0 tag | `Done` (2026-05-08) |
| [Phase 7](tasks/phase-7.md) | Deep Builder + LR Scheduling + Developer Skills — `pkg/optimizer/` scheduler extension, `pkg/nn` bulk constructors, `skills/gonn/` | `Done` (2026-05-10) |
| [Phase 8](tasks/phase-8.md) | LR Scheduling Extension + CLI Binary — `ExponentialLR`, `cmd/gonn/` binary, v0.7.0 release | `Done` (2026-05-10) |
| [Phase 9](archives/tasks/phase-9.md) | Metric Schedulers + Dynamic Topology + Observability Stack — `ReduceOnPlateau`, `OneCycleLR`, topology mutations, slog + viz HTTP server, v0.8.0 release | `Done (Archived)` (2026-05-10) |
| [Phase 10](archives/tasks/phase-10.md) | Normalization Layers + Training Callbacks — `pkg/layer/norm/` (BatchNorm/LayerNorm/GroupNorm), `pkg/nn/callbacks.go`, v0.9.0 release | `Done (Archived)` (2026-05-12) |
| [Phase 11](tasks/phase-11.md) | Meta-Learning Hooks + Convolutional Layers + Dataset Formats — `pkg/layer/conv/`, `pkg/dataset/mnist.go`, `pkg/nn/andtrain.go`, examples E06+E10, v0.10.0 release | `Done` (11/14 — Track A 3 tasks deferred to Phase 12; Track B+C+gate done; v0.10.0 tagged) |
| [Phase 12](archives/tasks/phase-12.md) | Meta-Learning Hooks — `pkg/nn/meta.go` (ParamAccessor[T]/ScalarParam[T]/SliceParam[T]/MetaLearner[T]), `WithMetaLearner` option, train.go hook, v0.11.0 release | `Done (Archived)` (5 tasks: T-12A01..A03 + T-12T01 + T-12Z01) |

## Phase 0 — Already Complete

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/activation/` | l2-activation-functions v1.0.0 | `Stable` |
| `pkg/loss/` | l2-loss-functions v1.0.0 | `Stable` |
| `pkg/utils/float.go`, `pkg/utils/logger.go` | utils baseline | `Stable` |

## Phase 1 Deliverables (Done)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/utils/errors.go` | l2-errors-impl v1.0.0 | `Stable` |
| `pkg/utils/init.go` | l2-init-impl v1.0.0 | `Stable` |
| `pkg/neuron/cell/*` | l2-neuron-model v1.1.0 | `Stable` |
| `pkg/neuron/axon/*` | l2-neuron-model v1.1.0 | `Stable` |
| `pkg/layer/*` | l2-layer-types v1.1.0 | `Stable` |
| `pkg/network/*` | l2-network-graph v1.1.0 | `Stable` |

## Phase 2 Deliverables (Done)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/nn/{config,nn,builder,compile}.go` | l2-nn-facade v2.0.0 | `Stable` |
| `pkg/nn/{options,presets}.go` | l2-nn-facade v2.0.0 | `Stable` |
| `pkg/nn/{train,query,verify}.go` | l2-training-loop v1.0.0 | `Stable` |
| `pkg/nn/control.go` | l2-control-impl v1.0.0 | `Stable` |

## Phase 3 Deliverables (Done)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/persistence/{config,weights}.go` | l2-persistence-impl v1.0.0 | `Stable` |
| `pkg/checkpoint/{snapshot,writer,reader,retention}.go` | l2-checkpointing-impl v1.0.0 | `Stable` |
| `pkg/dataset/{dataset,slice,csv,prefetch}.go` | l2-streaming-impl v1.0.0 | `Stable` |
| `pkg/compute/{backend,registry}.go` + `cpu/*` | l2-backend-cpu v1.0.0 | `Stable` |
| `pkg/network/{pool,worker}.go` + `pkg/nn/profiling.go` + bench files | l2-perf-impl v1.0.0 | `Stable` |

## Phase 4 Deliverables (Done — v0.5 scope)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `examples/xor/` (E01) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/style_showcase/` (E12) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/logic_gates/` (E02) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/callbacks/` (E11) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/persistence/` (E09) | l2-usage-examples v1.0.0 + l2-persistence-impl v1.0.0 | `Stable` |
| `examples/shared_options/` (E14, adapted) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/precision/` (E15) | l2-usage-examples v1.0.0 | `Stable` |
| `examples/README.md` + coverage audit | l2-usage-examples v1.0.0 §5.3 | `Stable` |

### Phase 4 v0.6 Promotion → Phase 5

Six of the eight v0.6-gated entries (E03, E04, E05, E07, E08, E13) are now active under Phase 5 — see deliverables table below.

## Phase 5 Deliverables (Done — 2026-05-06)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/network/{network,propagation}.go` (multi-hidden chain) | l2-multihidden-impl v1.0.0 | `Done` |
| `pkg/nn/compile.go` (gate lift + variadic SetLayers) | l2-multihidden-impl v1.0.0 | `Done` |
| `pkg/persistence/config.go` (SchemaVersion 1.1.0) | l2-multihidden-impl v1.0.0 §5.4 | `Done` |
| `examples/perceptron/` restored (E03) | l2-usage-examples v1.0.0 | `Done` |
| `examples/binary_classification/` (E04) | l2-usage-examples v1.0.0 | `Done` |
| `examples/iris/` (E05) | l2-usage-examples v1.0.0 | `Done` |
| `examples/regression_sin/` (E07) | l2-usage-examples v1.0.0 | `Done` |
| `examples/regression_multi/` (E08) | l2-usage-examples v1.0.0 | `Done` |
| `examples/higher_order_options/` (E13) | l2-usage-examples v1.0.0 | `Done` |

## Phase 6 Deliverables (Done — 2026-05-08)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/optimizer/` (SGD/Adam/RMSProp/Momentum) | l2-optimizer-impl v1.0.0 | `Done` |
| `pkg/regularizer/` (L1/L2/Dropout/Compose) | l2-regularization-impl v1.0.0 | `Done` |
| `pkg/neuron/axon/axon.go` (WeightInit debt fix) | l2-regularization-impl v1.0.0 T-6B06 | `Done` |
| `pkg/nn/options.go` + `compile.go` (WithOptimizer, WithRegularizer) | l2-optimizer-impl + l2-regularization-impl | `Done` |
| `pkg/nn/train.go` (opt.Step + reg.Penalty + reg.ApplyMask) | l2-training-loop v1.0.0 | `Done` |
| `CHANGELOG.md` (v0.6.0 entry) | l1-release-policy v1.0.0 | `Done` |
| `README.md` (v0.6 multi-hidden API docs) | l1-release-policy v1.0.0 | `Done` |
| `v0.6.0` git tag | l1-release-policy v1.0.0 §5.4 | `Done` |

## Phase 7 Deliverables (Done — 2026-05-10)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/optimizer/scheduler.go` (Scheduler[T] interface, BindScheduler, LearningRateSetter[T]) | l1-lr-scheduling v1.0.0 | `Done` |
| `pkg/optimizer/step_lr.go` | l1-lr-scheduling v1.0.0 | `Done` |
| `pkg/optimizer/warmup_lr.go` | l1-lr-scheduling v1.0.0 | `Done` |
| `pkg/optimizer/cosine_lr.go` | l1-lr-scheduling v1.0.0 | `Done` |
| `pkg/optimizer/chain_scheduler.go` | l1-lr-scheduling v1.0.0 | `Done` |
| `pkg/optimizer/scheduler_test.go` | l1-lr-scheduling v1.0.0 | `Done` |
| `pkg/nn/options.go` (WithScheduler, Repeat, Pattern, WithHiddenLayers) | l1-lr-scheduling + l2-deep-builder v1.0.0 | `Done` |
| `pkg/nn/builder.go` (Repeat, Pattern, HiddenLayers, WithScheduler) | l2-deep-builder v1.0.0 | `Done` |
| `pkg/nn/train.go` (scheduler dispatch per Granularity) | l1-lr-scheduling v1.0.0 | `Done` |
| `pkg/nn/config.go` (Scheduler field) | l1-lr-scheduling v1.0.0 | `Done` |
| `pkg/nn/phase7_test.go` | l2-deep-builder v1.0.0 | `Done` |
| `skills/gonn/SKILL.md` | l2-gonn-skills v1.0.0 | `Done` |
| `skills/gonn/examples/builder-xor.md` | l2-gonn-skills v1.0.0 | `Done` |
| `skills/gonn/examples/options-mnist.md` | l2-gonn-skills v1.0.0 | `Done` |
| `skills/gonn/examples/deep-network.md` | l2-gonn-skills v1.0.0 | `Done` |
| `skills/gonn/resources/api-reference.md` | l2-gonn-skills v1.0.0 | `Done` |
| `skills/gonn/resources/conventions.md` | l2-gonn-skills v1.0.0 | `Done` |

### Remaining v0.6 Backlog (gated on follow-up specs)

- **E06 MNIST preset** — awaits MNIST dataset-loader spec (planned post-Phase 5).
- **E10 Continuation** — awaits `AndTrain` API surface (planned post-Phase 5).
- **`PresetRegression` parameter extension** — v0.7 follow-up minor.

## Cross-Cutting Constraints

- **C29 Stdlib-only**: every new file uses Go standard library exclusively.
- **C30 Coverage floor**: every new package ships with ≥80% line coverage and `go test -race` clean.
- **C31 Doc-comments**: tier-by-audience verbosity (public > internal > unexported).
- **C32 Error informativeness**: every new error wraps a category sentinel from `pkg/utils/errors.go` (introduced in T-1A01).
- **C-001 Resolution**: Phase 1 closes the compilation blocker recorded in STATE.md. No integration testing until `go build ./...` is green.

## Meta Information

## Phase 8 Deliverables (Done — 2026-05-10)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/optimizer/exponential_lr.go` | l2-lr-scheduling-impl v1.1.0 | `Done` |
| `cmd/gonn/main.go` (routing) | l2-cli-client v0.2.0 | `Done` |
| `cmd/gonn/train.go` | l2-cli-client v0.2.0 §5.1.1 | `Done` |
| `cmd/gonn/query.go` + `verify.go` | l2-cli-client v0.2.0 §5.1.2–5.1.3 | `Done` |
| `cmd/gonn/version.go` + `exitcode.go` | l2-cli-client v0.2.0 §5.3 | `Done` |
| `cmd/gonn/csv.go` (streaming) | l2-cli-client v0.2.0 §5.2 | `Done` |
| `CHANGELOG.md` (v0.7.0 entry) | l1-release-policy v1.0.0 | `Done` |
| `v0.7.0` git tag | l1-release-policy v1.0.0 §5.4 | `Done` |

## Phase 9 Deliverables (Done — 2026-05-10)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/optimizer/metric_scheduler.go` | l2-metric-scheduler-impl v0.1.0 | `Done` |
| `pkg/optimizer/reduce_on_plateau.go` + test | l2-metric-scheduler-impl v0.1.0 §5.3 | `Done` |
| `pkg/optimizer/one_cycle_lr.go` + test | l2-metric-scheduler-impl v0.1.0 §5.4 | `Done` |
| `pkg/nn/train.go` (metric dispatch patch) | l2-metric-scheduler-impl v0.1.0 §5.5 | `Done` |
| `pkg/network/topology.go` | l2-dynamic-topology-impl v0.2.0 §5.1–5.6 | `Done` |
| `pkg/network/topology_test.go` | l2-dynamic-topology-impl v0.2.0 | `Done` |
| `pkg/utils/errors.go` (new DYN sentinels) | l2-dynamic-topology-impl v0.2.0 §5.6 | `Done` |
| `pkg/nn/options.go` + `builder.go` (WithTopologyMode) | l2-dynamic-topology-impl v0.2.0 §5.1 | `Done` |
| `pkg/utils/logger.go` (slog upgrade, LevelTrace) | l2-logging-strategy v0.2.0 §5.5 | `Done` |
| `pkg/visualization/server.go` + `handlers.go` + `middleware.go` | l2-visualization-api v0.2.0 §5.5–5.6 | `Done` |
| `pkg/nn/options.go` (WithLogger, WithVisualizationEndpoint) + `compile.go` | l2-visualization-api + l2-logging-strategy | `Done` |
| `CHANGELOG.md` (v0.8.0 entry) | l1-release-policy v1.0.0 | `Done` |
| `v0.8.0` git tag | l1-release-policy v1.0.0 §5.4 | `Done` |

## Phase 10 Deliverables (Done — 2026-05-12)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/layer/norm/norm.go` (interface + helpers) | l2-normalization-impl v0.1.0 | `Done` |
| `pkg/layer/norm/batchnorm.go` + `norm_test.go` | l2-normalization-impl v0.1.0 §5.2 | `Done` |
| `pkg/layer/norm/layernorm.go` + `groupnorm.go` | l2-normalization-impl v0.1.0 §5.2 | `Done` |
| `pkg/nn/options.go` (WithBatchNorm/WithLayerNorm/WithNormAfterLayer) | l2-normalization-impl v0.1.0 §5.3 | `Done` |
| `pkg/nn/compile.go` (norm layer injection + grad slots) | l2-normalization-impl v0.1.0 §6 | `Done` |
| `pkg/nn/nn.go` (SetTrain/SetEval propagation) | l2-normalization-impl v0.1.0 §5.4 | `Done` |
| `pkg/nn/callbacks.go` (CallbackRegistry[T], ErrStopTraining, fireEvent) | l2-callbacks-impl v0.1.0 | `Done` |
| `pkg/nn/callbacks_test.go` (BenchmarkNoCallbacks) | l2-callbacks-impl v0.1.0 §5.1 | `Done` |
| `pkg/nn/train.go` (defer fireOnTrainEnd + dispatch) | l2-callbacks-impl v0.1.0 §5.4 | `Done` |
| `pkg/nn/options.go` (WithOnIterationEnd/WithOnImprovementFound/WithOnTrainEnd) | l2-callbacks-impl v0.1.0 §5.3 | `Done` |
| `pkg/utils/errors.go` (ErrCallbackPanic sentinel) | l2-callbacks-impl v0.1.0 §6 | `Done` |
| `CHANGELOG.md` (v0.9.0 entry) | l1-release-policy v1.0.0 | `Done` |
| `v0.9.0` git tag | l1-release-policy v1.0.0 §5.4 | `Done` |

## Phase 11 Deliverables (Done)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/nn/meta.go` (ParamAccessor[T], ScalarParam[T], SliceParam[T], MetaLearner[T]) | l2-meta-learning-impl v0.1.0 §5.2 | `Done (Phase 12)` |
| `pkg/nn/options.go` (WithMetaLearner) + `config.go` | l2-meta-learning-impl v0.1.0 §5.5 | `Done (Phase 12)` |
| `pkg/nn/train.go` (meta hook after opt.Step) | l2-meta-learning-impl v0.1.0 §5.3 | `Done (Phase 12)` |
| `pkg/utils/errors.go` (ErrMetaLearnerShape, ErrMetaLearnerRunning) | l2-meta-learning-impl v0.1.0 §5.4 | `Done (Phase 12)` |
| `pkg/layer/conv/conv.go` (interface assertions) | l2-conv-layers-impl v0.1.0 §5.1 | `Done` [B] |
| `pkg/layer/conv/conv1d.go` (Conv1D[T]) | l2-conv-layers-impl v0.1.0 §5.2 | `Done` [B] |
| `pkg/layer/conv/pool.go` (MaxPool1D[T], AvgPool1D[T]) | l2-conv-layers-impl v0.1.0 §5.3 | `Done` [B] |
| `pkg/layer/conv/flatten.go` (Flatten[T]) | l2-conv-layers-impl v0.1.0 §5.4 | `Done` [B] |
| `pkg/nn/options.go` (WithConv1D, WithMaxPool1D, WithAvgPool1D, WithFlatten) | l2-conv-layers-impl v0.1.0 §5.5 | `Done` [B] |
| `pkg/nn/compile.go` (conv stack prepend + outputLen) | l2-conv-layers-impl v0.1.0 §5.6 | `Done` [B] |
| `pkg/utils/errors.go` (ErrConvShapeMismatch, ErrConvPoolSizeMismatch) | l2-conv-layers-impl v0.1.0 §5.7 | `Done` [B] |
| `pkg/network/propagation.go` (AppendInputGradient) | l2-conv-layers-impl v0.1.0 §5.8 | `Done` [B] |
| `pkg/dataset/mnist.go` (IDXReader, MNISTLoader[T]) | l2-dataset-loader-impl v0.1.0 §5.2–5.3 | `Done` [C] |
| `pkg/nn/andtrain.go` (AndTrain method) | l2-dataset-loader-impl v0.1.0 §5.4 | `Done` [C] |
| `pkg/utils/errors.go` (ErrIDXMagic, ErrMNISTRecordMismatch, ErrNetworkRunning) | l2-dataset-loader-impl v0.1.0 §5.5 | `Done` [C] |
| `examples/mnist/` (E06 — smoke deferred: IDX data not committed) | l2-usage-examples v1.0.0 + l2-dataset-loader-impl v0.1.0 | `Done` [C] |
| `examples/continuation/` (E10) | l2-usage-examples v1.0.0 + l2-dataset-loader-impl v0.1.0 | `Done` [C] |
| `CHANGELOG.md` (v0.10.0 entry) | l1-release-policy v1.0.0 | `Done` |
| `v0.10.0` git tag | l1-release-policy v1.0.0 §5.4 | `Pending user` (suggested `git tag -a v0.10.0`) |

## Phase 12 Deliverables (Done — Archived)

| Asset | Spec | Status |
| :--- | :--- | :--- |
| `pkg/nn/meta.go` (ParamAccessor[T], ScalarParam[T], SliceParam[T], MetaLearner[T], FeatureFunc[T]) | l2-meta-learning-impl v0.1.0 §5.2 | `Done` [A] |
| `pkg/nn/options.go` (WithMetaLearner) | l2-meta-learning-impl v0.1.0 §5.5 | `Done` [A] |
| `pkg/nn/config.go` (MetaLearner field) | l2-meta-learning-impl v0.1.0 §5.5 | `Done` [A] |
| `pkg/nn/train.go` (meta hook after opt.Step, before OnIterationEnd) | l2-meta-learning-impl v0.1.0 §5.3 | `Done` [A] |
| `pkg/utils/errors.go` (ErrMetaLearnerShape, ErrMetaLearnerRunning) | l2-meta-learning-impl v0.1.0 §5.4 | `Done` [A] |
| `pkg/nn/meta_test.go` (14 tests: step variants + wiring + convergence) | l2-meta-learning-impl v0.1.0 §5.3 + §5.5 | `Done` [T] |
| `CHANGELOG.md` (v0.11.0 entry) | l1-release-policy v1.0.0 | `Done` [Z] |
| `v0.11.0` git tag | l1-release-policy v1.0.0 §5.4 | `Pending user` (run `git tag -a v0.11.0`) |

- **Last Updated**: 2026-05-17 (Phase 12 gate passed: `go build ./...` clean; 19/19 packages green; `pkg/nn` 82.2 % coverage; CHANGELOG v0.11.0 written; phase-12.md archived; v0.11.0 tag pending user. TASKS.md v2.7.0 → v2.8.0.)
- **Maintainer**: Core Team
