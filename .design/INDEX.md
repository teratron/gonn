# Workspace Specifications Registry

**Version:** 1.5.0
**Status:** Active

## Overview

Local registry of specifications for this workspace.

## Domain Specifications

| File | Description | Status | Layer | Version |
| :--- | :--- | :--- | :--- | :--- |
| [l1-neural-network-architecture.md](specifications/l1-neural-network-architecture.md) | Conceptual architecture of the neural network library | Stable | L1 | 1.0.0 |
| [l1-math-functions.md](specifications/l1-math-functions.md) | Mathematical functions framework (activation and loss) | Stable | L1 | 1.0.0 |
| [l1-training-semantics.md](specifications/l1-training-semantics.md) | `Train()` convergence loop contract — max-iter, loss-limit, min-loss rollback | RFC | L1 | 0.1.0 |
| [l1-weight-initialization.md](specifications/l1-weight-initialization.md) | Xavier / He / Random formulas + RNG seed contract | RFC | L1 | 0.1.0 |
| [l1-network-persistence.md](specifications/l1-network-persistence.md) | JSON schema for config and weights, round-trip integrity | RFC | L1 | 0.1.0 |
| [l1-training-control.md](specifications/l1-training-control.md) | Training lifecycle state machine (Pause/Resume/Stop) | Draft | L1 | 0.1.0 |
| [l1-checkpointing.md](specifications/l1-checkpointing.md) | Snapshot/recovery system with retention and compression | Draft | L1 | 0.1.0 |
| [l1-dynamic-topology.md](specifications/l1-dynamic-topology.md) | Dynamic mode for runtime layer/neuron mutation (amends INV-2) | Draft | L1 | 0.1.0 |
| [l1-observability-protocol.md](specifications/l1-observability-protocol.md) | Read-only state-exposure contract for external observers | Draft | L1 | 0.1.0 |
| [l1-error-taxonomy.md](specifications/l1-error-taxonomy.md) | Canonical error categories and wrapping contract | RFC | L1 | 0.1.0 |
| [l1-performance-contract.md](specifications/l1-performance-contract.md) | Performance guarantees, benchmarks, optimization layers | Draft | L1 | 0.1.0 |
| [l1-data-streaming.md](specifications/l1-data-streaming.md) | Streaming dataset abstraction and bounded memory contract | Draft | L1 | 0.1.0 |
| [l1-compute-backend.md](specifications/l1-compute-backend.md) | Pluggable compute backend (CPU baseline + GPU future) | Draft | L1 | 0.1.0 |
| [l1-meta-learning-hooks.md](specifications/l1-meta-learning-hooks.md) | Adaptive parameter tuning via inner-network recursion | Draft | L1 | 0.1.0 |
| [l2-nn-facade.md](specifications/l2-nn-facade.md) | Public API facade — dual-style fluent API (Builder + Functional Options) | RFC | L2 | 2.0.0 |
| [l2-network-graph.md](specifications/l2-network-graph.md) | Internal computational graph — Network[T] and bundles | Stable | L2 | 1.0.0 |
| [l2-layer-types.md](specifications/l2-layer-types.md) | Layer type hierarchy — Input, Dense, Output | Stable | L2 | 1.0.0 |
| [l2-neuron-model.md](specifications/l2-neuron-model.md) | Neuron/Cell/Axon model and interfaces | Stable | L2 | 1.0.0 |
| [l2-activation-functions.md](specifications/l2-activation-functions.md) | 10 activation functions with dispatcher pattern | Stable | L2 | 1.0.0 |
| [l2-loss-functions.md](specifications/l2-loss-functions.md) | 18 loss functions with dispatcher pattern | Stable | L2 | 1.0.0 |
| [l2-usage-examples.md](specifications/l2-usage-examples.md) | Canonical example catalog — 15 entries with coverage matrix | Draft | L2 | 1.0.0 |
| [l2-cli-client.md](specifications/l2-cli-client.md) | CLI binary `gonn` — train/query/verify subcommands | Draft | L2 | 0.1.0 |
| [l2-visualization-api.md](specifications/l2-visualization-api.md) | HTTP/JSON adapter exposing observability for external GUIs | Draft | L2 | 0.1.0 |
| [l2-logging-strategy.md](specifications/l2-logging-strategy.md) | Structured logging via `log/slog` (Trace/Debug/Info/Warn/Error) | Draft | L2 | 0.1.0 |

## Meta Information

- **Maintainer**: Core Team
- **Last Updated**: 2026-04-27
