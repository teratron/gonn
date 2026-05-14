# Neuron Model

**Version:** 1.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-neural-network-architecture.md

## Overview

Specifies the neuron (cell) model and axon (connection) types that form the computational units of the neural network. The model uses interface segregation: `Nucleus[T]` for read-only value access and `Neuron[T]` for full computation capability. Cells are connected via axons that carry weighted values.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent concept spec
- [l2-network-graph.md](l2-network-graph.md) — Network containing cell bundles
- [l2-layer-types.md](l2-layer-types.md) — Layers that manage cell collections

## 1. Motivation

The neuron model is the fundamental computational unit. By separating the interface into read-only (`Nucleus`) and compute (`Neuron`), the system enforces that input cells (which only store values) cannot be accidentally used for computation, while dense and output cells have full forward/backward capabilities.

## 2. Constraints & Assumptions

- All cell types are generic over `utils.Float`
- Axon weights initialized randomly in range [-0.5, 0.5] with mutex protection
- Cell identity encoded as `[2]uint` (layer index, cell index)
- Compile-time interface verification is mandatory for all cell types

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| INV-1 (Generic Float) | All types parameterized by `T utils.Float` |
| INV-6 (Interface segregation) | `Nucleus[T]` (GetValue) vs `Neuron[T]` (GetValue + Miss + Calculate) |
| INV-8 (Axon-only communication) | `Axon[T]` stores `Cell Nucleus[T]` reference for incoming value |
| INV-3 (Forward propagation) | `axon.CalculateValue()` computes `cell.value * weight` |
| INV-4 (Backward propagation) | `axon.CalculateMiss()` uses `OutgoingCell` for backprop (currently broken) |
| INV-5 (Gradient descent) | `axon.CalculateWeight(gradient)` updates weight |

## 5. Detailed Design

### 5.1 Interface Hierarchy

```plaintext
Nucleus[T utils.Float]  (interface)
├── GetValue() *T

Neuron[T utils.Float]  (interface, extends Nucleus)
├── GetValue() *T
├── GetMiss() *T
├── SetMiss(T)
├── CalculateValue()
└── CalculateWeight(*T)
```

### 5.2 Cell Types

```plaintext
core[T]  (unexported)
├── Id     [2]uint
├── value  T
├── Implements: Nucleus[T]

Input[T] = core[T]
├── SetValue(T)
├── Implements: Nucleus[T]

Bias[T] = core[T]
├── value always 1.0
├── Implements: Nucleus[T]

Dense[T]  (embeds *core[T])
├── miss   T
├── Axons  axon.Bundle[T]
├── Implements: Neuron[T]
├── CalculateValue(): sum(axon.values) → activation
├── CalculateWeight(gradient): update all axon weights

Output[T]  (embeds *Dense[T])
├── target  *T
├── Implements: Neuron[T]
├── CalculateValue(): Dense.CalculateValue() (BROKEN: infinite recursion)

Hidden[T]  — DOES NOT EXIST (referenced in network.go but undefined)
```

### 5.3 Axon Model

```plaintext
Axon[T utils.Float]
├── Weight  T                    (randomly initialized [-0.5, 0.5])
├── Cell    neuron.Nucleus[T]    (incoming cell reference)
├── OutgoingCell  (COMMENTED OUT — breaks backward propagation)
├── CalculateValue() T           (cell.value * weight)
├── CalculateMiss() T            (BROKEN: uses OutgoingCell)
├── CalculateWeight(gradient T)  (weight += gradient * cell.value)

Bundle[T] = []*Axon[T]
```

### 5.4 Known Issues (Critical)

1. **`cell.Hidden[T]` does not exist** — must be created (type alias to Dense or new type)
2. **`Output.CalculateValue()` infinite recursion** — calls `o.CalculateValue()` instead of `o.Dense.CalculateValue()`
3. **`axon.OutgoingCell` commented out** — backward propagation cannot function
4. **`axon.New()` ignores `outgoingCell` parameter** — silently discards it
5. **`_NewBias` with underscore** — unexported and unused
6. **Random init uses deprecated API** — `rand.New(rand.NewSource(time.Now().UnixNano()))`

## 6. Implementation Notes

1. Create `Hidden[T]` type (recommended: type alias to `Dense[T]`)
2. Fix `Output.CalculateValue()` to call embedded Dense method
3. Restore `OutgoingCell` field in Axon and `New()` constructor
4. Replace deprecated random initialization with `rand.New(rand.NewPCG(...))`
5. Add compile-time interface checks for Hidden[T]

## 7. Drawbacks & Alternatives

- **Alternative**: Single `Cell` interface without segregation — simpler but less safe
- **Alternative**: Store both incoming and outgoing cells in Axon — adds memory but simplifies backprop
- **Drawback**: Embedding chain (`Output → Dense → core`) creates deep nesting

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[NEURON]` | `pkg/neuron/neuron.go` | Interface definitions and type constants |
| `[CORE_CELL]` | `pkg/neuron/cell/core.go` | Base cell type |
| `[INPUT_CELL]` | `pkg/neuron/cell/input.go` | Input cell |
| `[DENSE_CELL]` | `pkg/neuron/cell/dense.go` | Dense cell with axons |
| `[OUTPUT_CELL]` | `pkg/neuron/cell/output.go` | Output cell with target |
| `[BIAS_CELL]` | `pkg/neuron/cell/bias.go` | Bias cell (constant 1.0) |
| `[AXON]` | `pkg/neuron/axon/axon.go` | Axon connection type |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-04-21 | Initial Stable — reverse-engineered from existing codebase |
| 1.1.0 | 2026-04-28 | Re-confirmed Stable under parent INV-2 v2.0 (TopologyMode). Neuron/Cell/Axon model is mode-agnostic — Dynamic-mode mutations operate on bundles, not on cells. No material change to interfaces. |
