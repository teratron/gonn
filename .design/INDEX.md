# Workspace Specifications Registry

**Version:** 2.3.0
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
| [l1-dynamic-topology.md](specifications/l1-dynamic-topology.md) | Dynamic mode — layer/neuron mutation lifecycle with transaction protocol | Draft | L1 | 0.2.0 |
| [l1-meta-learning-hooks.md](specifications/l1-meta-learning-hooks.md) | Universal parameter access for recursive self-optimization via inner-network | Draft | L1 | 0.2.0 |
| [l2-nn-facade.md](specifications/l2-nn-facade.md) | Public API facade — dual-style fluent API (Builder + Functional Options) | Stable | L2 | 2.0.0 |
| [l2-network-graph.md](specifications/l2-network-graph.md) | Internal computational graph — Network[T] and bundles | Stable | L2 | 1.1.0 |
| [l2-layer-types.md](specifications/l2-layer-types.md) | Layer type hierarchy — Input, Dense, Output | Stable | L2 | 1.1.0 |
| [l2-neuron-model.md](specifications/l2-neuron-model.md) | Neuron/Cell/Axon model and interfaces | Stable | L2 | 1.1.0 |
| [l2-activation-functions.md](specifications/l2-activation-functions.md) | 10 activation functions with dispatcher pattern | Stable | L2 | 1.0.0 |
| [l2-loss-functions.md](specifications/l2-loss-functions.md) | 18 loss functions with dispatcher pattern | Stable | L2 | 1.0.0 |
| [l2-usage-examples.md](specifications/l2-usage-examples.md) | Canonical example catalog — 15 entries with coverage matrix | Stable | L2 | 1.0.0 |
| [l2-cli-client.md](specifications/l2-cli-client.md) | CLI binary `gonn` — train/query/verify subcommands | RFC | L2 | 0.1.0 |
| [l2-visualization-api.md](specifications/l2-visualization-api.md) | HTTP/JSON adapter exposing observability for external GUIs | Draft | L2 | 0.1.0 |
| [l2-logging-strategy.md](specifications/l2-logging-strategy.md) | Structured logging via `log/slog` (Trace/Debug/Info/Warn/Error) | Draft | L2 | 0.1.0 |
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

## Meta Information

- **Maintainer**: Core Team
- **Last Updated**: 2026-05-03 (l2-multihidden-impl promoted Draft → Stable v1.0.0 for Phase 5 decomposition)
