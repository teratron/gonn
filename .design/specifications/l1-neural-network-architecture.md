# Neural Network Architecture

**Version:** 1.0.0
**Status:** Stable
**Layer:** concept

## Overview

Defines the conceptual architecture of the GoNN neural network library. GoNN provides a generic, type-safe neural network implementation in Go using generics (`float32|float64`). The architecture follows a layered composition model: Network contains Layers, Layers contain Cells (neurons), and Cells are connected via Axons (weighted connections).

## Related Specifications

- [l1-math-functions.md](l1-math-functions.md) — Mathematical functions used by neurons and loss computation
- [l2-nn-facade.md](l2-nn-facade.md) — Public API facade (implements this spec)
- [l2-network-graph.md](l2-network-graph.md) — Computational graph (implements this spec)
- [l2-layer-types.md](l2-layer-types.md) — Layer type hierarchy (implements this spec)
- [l2-neuron-model.md](l2-neuron-model.md) — Neuron/Cell/Axon model (implements this spec)

## 1. Motivation

Go lacks a native, zero-dependency neural network library that leverages Go generics for type safety. GoNN fills this gap by providing a composable, generic neural network library using only the standard library. The architecture must support:

- Type-safe numeric computation via Go generics
- Fluent builder API for network construction
- Forward and backward propagation
- Extensible activation and loss function sets

## 2. Constraints & Assumptions

- Zero external dependencies — standard library only
- Generic type parameter constrained to `float32 | float64` via `utils.Float`
- Single-threaded training loop (concurrency reserved for future phases)
- Feedforward topology only (no recurrent/convolutional layers in initial scope)

## 3. Core Invariants

- **INV-1**: All numeric computation must be generic over `utils.Float` constraint. No hardcoded float types.
- **INV-2**: Network topology is immutable after construction (Build phase). Runtime modification is not permitted.
- **INV-3**: Forward propagation computes values layer-by-layer from Input to Output. Each cell computes the weighted sum of its incoming axon values plus activation.
- **INV-4**: Backward propagation computes loss gradients from Output to Input. Each cell propagates its miss (error) backward through outgoing axons.
- **INV-5**: Weight updates use gradient descent with a configurable learning rate.
- **INV-6**: Interface segregation: read-only access (`Nucleus[T]`) is separated from compute access (`Neuron[T]`).
- **INV-7**: Layer composition follows `core → base → specialized` embedding hierarchy. Each level adds capabilities without modifying the parent.
- **INV-8**: Axon connections are the sole mechanism for inter-cell communication. No direct cell-to-cell references outside the axon graph.

## 5. Detailed Design

### 5.1 Architecture Overview

```mermaid
graph TD
    NN["NN[T] (Facade)"] -->|embeds| NET["Network[T]"]
    NET -->|contains| IB["Input Bundle"]
    NET -->|contains| HB["Hidden Bundles"]
    NET -->|contains| OB["Output Bundle"]
    IB -->|contains| IC["Input Cells"]
    HB -->|contains| DC["Dense Cells"]
    OB -->|contains| OC["Output Cells"]
    DC -->|has| AX["Axon Bundle"]
    OC -->|has| AX2["Axon Bundle"]
    AX -->|references| IC
    AX2 -->|references| DC
```

### 5.2 Data Flow

```mermaid
graph LR
    IN["Input Data"] --> IC["Input Cells"]
    IC -->|axons| DC["Dense Cells"]
    DC -->|activation| DC
    DC -->|axons| OC["Output Cells"]
    OC -->|activation| OUT["Output Values"]
    OUT -->|loss| LOSS["Loss Computation"]
    LOSS -->|backprop| OC
    OC -->|miss| DC
    DC -->|weight update| AX["Axons"]
```

### 5.3 Generic Type System

All numeric types are parameterized by `T utils.Float` where `Float` is defined as `float32 | float64`. This constraint propagates through the entire type hierarchy: `NN[T]`, `Network[T]`, `bundle[T,S]`, cell types, and `Axon[T]`.

### 5.4 Builder Pattern

Network construction uses a fluent builder:
- `New[T]()` creates an empty network
- `.Input(size)` adds the input layer
- `.Dense(size, activation, bias)` adds hidden layers
- `.Output(size, activation, loss, bias)` adds the output layer

The builder returns `*NN[T]` at each step, enabling method chaining.

## 6. Implementation Notes

1. Resolve compilation blockers first (Hidden cell type, OutgoingCell field, Output recursion)
2. Complete Build() to wire all layer connections
3. Implement Train/Query/Verify methods in the facade
4. Add comprehensive test coverage (target: 80%)

## 7. Drawbacks & Alternatives

- **Alternative**: Use reflection instead of generics — rejected for performance and type safety
- **Alternative**: Use external math libraries (gonum) — rejected to maintain zero-dependency policy
- **Drawback**: Go generics lack higher-kinded types, limiting some abstraction patterns
- **Drawback**: Slice type assertions in Go are not covariant, requiring element-by-element iteration

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[NN]` | `pkg/nn/nn.go` | Facade type definition and builder methods |
| `[NETWORK]` | `pkg/network/network.go` | Core computational graph structure |
| `[NEURON]` | `pkg/neuron/neuron.go` | Interface definitions (Nucleus, Neuron) |
| `[FLOAT]` | `pkg/utils/float.go` | Generic type constraint |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-04-21 | Initial Stable — reverse-engineered from existing codebase |
