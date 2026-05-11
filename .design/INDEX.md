# Workspace Specifications Registry

**Version:** 2.6.0
**Status:** Active

## Overview

Local registry of specifications for this workspace.

## Domain Specifications

| File | Description | Status | Layer | Version |
| :--- | :--- | :--- | :--- | :--- |
| [l1-neural-network-architecture.md](specifications/l1-neural-network-architecture.md) | Conceptual architecture — INV-2 amended with TopologyMode (v2.0) | Stable | L1 | 2.0.0 |
| [l1-math-functions.md](specifications/l1-math-functions.md) | Mathematical functions framework (activation and loss) | Stable | L1 | 1.0.0 |
| [l1-training-semantics.md](specifications/l1-training-semantics.md) | `Train()` convergence loop contract — max-iter, loss-limit, min-loss rollback | Stable | L1 | 1.0.0 |
| [l1-weight-initialization.md](specifications/l1-weight-initialization.md) | Xavier / He / Random formulas + RNG seed contract | Stable | L1 | 1.0.0 |
| [l1-network-persistence.md](specifications/l1-network-persistence.md) | JSON schema for config and weights, round-trip integrity | Stable | L1 | 1.0.0 |
| [l1-training-control.md](specifications/l1-training-control.md) | Training lifecycle state machine (Pause/Resume/Stop) | Stable | L1 | 1.0.0 |
| [l1-checkpointing.md](specifications/l1-checkpointing.md) | Snapshot/recovery system with retention and compression | Stable | L1 | 1.0.0 |
| [l1-error-taxonomy.md](specifications/l1-error-taxonomy.md) | Canonical error categories and wrapping contract | Stable | L1 | 1.0.0 |
| [l1-observability-protocol.md](specifications/l1-observability-protocol.md) | Read-only state-exposure contract for external observers | Stable | L1 | 1.0.0 |
| [l1-performance-contract.md](specifications/l1-performance-contract.md) | Performance guarantees, benchmarks, optimization layers | Stable | L1 | 1.0.0 |
| [l1-data-streaming.md](specifications/l1-data-streaming.md) | Streaming dataset abstraction and bounded memory contract | Stable | L1 | 1.0.0 |
| [l1-compute-backend.md](specifications/l1-compute-backend.md) | Pluggable compute backend (CPU baseline + GPU future) | Stable | L1 | 1.0.0 |
| [l1-dynamic-topology.md](specifications/l1-dynamic-topology.md) | Dynamic mode — layer/neuron mutation lifecycle with transaction protocol | Stable | L1 | 0.2.0 |
| [l1-meta-learning-hooks.md](specifications/l1-meta-learning-hooks.md) | Universal parameter access for recursive self-optimization via inner-network | RFC | L1 | 0.3.0 |
| [l2-nn-facade.md](specifications/l2-nn-facade.md) | Public API facade — dual-style fluent API (Builder + Functional Options) | Stable | L2 | 2.0.0 |
| [l2-network-graph.md](specifications/l2-network-graph.md) | Internal computational graph — Network[T] and bundles | Stable | L2 | 1.1.0 |
| [l2-layer-types.md](specifications/l2-layer-types.md) | Layer type hierarchy — Input, Dense, Output | Stable | L2 | 1.1.0 |
| [l2-neuron-model.md](specifications/l2-neuron-model.md) | Neuron/Cell/Axon model and interfaces | Stable | L2 | 1.1.0 |
| [l2-activation-functions.md](specifications/l2-activation-functions.md) | 10 activation functions with dispatcher pattern | Stable | L2 | 1.0.0 |
| [l2-loss-functions.md](specifications/l2-loss-functions.md) | 18 loss functions with dispatcher pattern | Stable | L2 | 1.0.0 |
| [l2-usage-examples.md](specifications/l2-usage-examples.md) | Canonical example catalog — 15 entries with coverage matrix | Stable | L2 | 1.0.0 |
| [l2-cli-client.md](specifications/l2-cli-client.md) | CLI binary `gonn` — train/query/verify subcommands | Stable | L2 | 0.2.0 |
| [l2-visualization-api.md](specifications/l2-visualization-api.md) | HTTP/JSON adapter exposing observability for external GUIs — pkg/visualization/ package structure | Stable | L2 | 0.2.0 |
| [l2-logging-strategy.md](specifications/l2-logging-strategy.md) | Structured logging via `log/slog` — goLogger wrapper, LevelTrace, WithLogger option | Stable | L2 | 0.2.0 |
| [l2-training-loop.md](specifications/l2-training-loop.md) | Go realization of training-semantics — `Train()` body and snapshot mechanics | Stable | L2 | 1.0.0 |
| [l2-persistence-impl.md](specifications/l2-persistence-impl.md) | Go realization of persistence — `pkg/persistence` package | Stable | L2 | 1.0.0 |
| [l2-errors-impl.md](specifications/l2-errors-impl.md) | Go realization of error taxonomy — sentinels and helper constructors | Stable | L2 | 1.0.0 |
| [l2-init-impl.md](specifications/l2-init-impl.md) | Go realization of weight init — sampling helpers + RNG plumbing | Stable | L2 | 1.0.0 |
| [l2-control-impl.md](specifications/l2-control-impl.md) | Go realization of training control — atomic state cell + safe-point check | Stable | L2 | 1.0.0 |
| [l2-checkpointing-impl.md](specifications/l2-checkpointing-impl.md) | Go realization of checkpointing — `pkg/checkpoint` package, atomic write, retention | Stable | L2 | 1.0.0 |
| [l2-perf-impl.md](specifications/l2-perf-impl.md) | Go realization of performance contract — pools, preallocation, pprof hook | Stable | L2 | 1.0.0 |
| [l2-streaming-impl.md](specifications/l2-streaming-impl.md) | Go realization of data streaming — `pkg/dataset` package, prefetch decorator | Stable | L2 | 1.0.0 |
| [l2-backend-cpu.md](specifications/l2-backend-cpu.md) | Go realization of compute backend — `pkg/compute/cpu` reference path | Stable | L2 | 1.0.0 |
| [l2-multihidden-impl.md](specifications/l2-multihidden-impl.md) | v0.6 multi-hidden topology lift — Network[T] chain + compile() gate removal | Stable | L2 | 1.0.0 |
| [l1-optimizer-strategies.md](specifications/l1-optimizer-strategies.md) | Optimizer strategies contract — SGD / Adam / RMSProp / Momentum | Stable | L1 | 1.0.0 |
| [l2-optimizer-impl.md](specifications/l2-optimizer-impl.md) | Go realization of optimizer strategies — pkg/optimizer package | Stable | L2 | 1.0.0 |
| [l1-regularization.md](specifications/l1-regularization.md) | Regularization contract — L1 / L2 / Dropout / Compose | Stable | L1 | 1.0.0 |
| [l2-regularization-impl.md](specifications/l2-regularization-impl.md) | Go realization of regularization — pkg/regularizer package | Stable | L2 | 1.0.0 |
| [l1-release-policy.md](specifications/l1-release-policy.md) | Semantic versioning and release gate contract for GoNN | Stable | L1 | 1.0.0 |
| [l2-ai-doc-metadata.md](specifications/l2-ai-doc-metadata.md) | AI-Meta trailing block for doc comments — closed vocabulary, tier-gated, process-artifact firewall | RFC | L2 | 0.1.0 |
| [l1-lr-scheduling.md](specifications/l1-lr-scheduling.md) | Learning rate scheduler contract — StepLR / CosineAnnealing / WarmUp / ChainScheduler | Stable | L1 | 1.0.0 |
| [l2-deep-builder.md](specifications/l2-deep-builder.md) | Deep network builder ergonomics — Repeat / Pattern / HiddenLayers bulk constructors | Stable | L2 | 1.0.0 |
| [l2-gonn-skills.md](specifications/l2-gonn-skills.md) | GoNN developer AI skills — SKILL.md for AI-assisted code generation | Stable | L2 | 1.0.0 |
| [l2-lr-scheduling-impl.md](specifications/l2-lr-scheduling-impl.md) | Go realization of LR scheduling — pkg/optimizer/ schedulers (Phase 7+8) | Stable | L2 | 1.1.0 |
| [l2-dynamic-topology-impl.md](specifications/l2-dynamic-topology-impl.md) | Go realization of dynamic topology — mutation methods on Network[T] | Stable | L2 | 0.2.0 |
| [l2-metric-scheduler-impl.md](specifications/l2-metric-scheduler-impl.md) | Go realization of metric-driven LR schedulers — MetricScheduler[T], ReduceOnPlateau[T], OneCycleLR[T] | Stable | L2 | 0.1.0 |
| [l1-normalization-layers.md](specifications/l1-normalization-layers.md) | Normalization layer contract — BatchNorm / LayerNorm / GroupNorm with affine params and train/eval mode | Stable | L1 | 1.0.0 |
| [l1-training-callbacks.md](specifications/l1-training-callbacks.md) | Training event callbacks — OnIterationEnd / OnImprovementFound / OnTrainEnd with StopTraining signal | Stable | L1 | 1.0.0 |

## Meta Information

- **Maintainer**: Core Team
- **Last Updated**: 2026-05-11 (3 dispatches: l1-normalization-layers Stable v1.0.0 (NORM-1..9), l1-training-callbacks Stable v1.0.0 (CB-1..9, 3 events), l1-meta-learning-hooks Draft→RFC v0.3.0 (8 TBDs closed).)
