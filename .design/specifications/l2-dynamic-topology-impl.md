# Dynamic Topology Implementation

**Version:** 0.1.0
**Status:** Draft
**Layer:** implementation
**Implements:** l1-dynamic-topology.md

## Overview

Go realization of the Dynamic Topology contract (`l1-dynamic-topology.md`) in `pkg/network/` and
`pkg/nn/`. Implements opt-in `TopologyMode` on `Network[T]`, mutation methods
(`AddHiddenLayer`, `RemoveHiddenLayer`, `AddNeuron`, `RemoveNeuron`), transactional rebalance,
and `TopologyVersion` counter exposure.

This spec is the blueprint for v0.7.0 Phase A implementation. No code exists yet.

## Related Specifications

- [l1-dynamic-topology.md](l1-dynamic-topology.md) — Parent contract (DYN-1..DYN-6)
- [l2-network-graph.md](l2-network-graph.md) — `Network[T]` that receives mutation methods
- [l2-layer-types.md](l2-layer-types.md) — Layer types subject to mutation
- [l2-neuron-model.md](l2-neuron-model.md) — Cell/Axon model re-wired during mutations
- [l2-control-impl.md](l2-control-impl.md) — Atomic state cell guarding Paused/Idle precondition
- [l2-checkpointing-impl.md](l2-checkpointing-impl.md) — Must persist TopologyVersion in snapshots
- [l2-init-impl.md](l2-init-impl.md) — Weight initializer for new connections after mutations
- [l2-errors-impl.md](l2-errors-impl.md) — New sentinel errors for DYN error cases

## 1. Motivation

Phase 7 delivered deep builders (100+ layer construction). The next natural step is runtime
topology mutation: growing a network during training when loss plateaus, pruning dead neurons,
or progressively deepening shallow networks without full rebuild. The L1 contract is Stable at
v0.2.0; this spec realizes it in Go.

## 2. Constraints & Assumptions

- Dynamic mode is opt-in (`WithTopologyMode(Dynamic)`); Immutable remains the default (DYN-1).
- Mutations are gated on `Paused` or `Idle` state via `pkg/network/state.go` atomic cell (DYN-2).
- `pkg/layer/` and `pkg/neuron/cell/` provide the layer/cell constructors used during mutation.
- Input and Output layers are always index 0 and `len(layers)-1`; hidden layers are indices 1..n-2.
- Transaction rollback re-applies a shallow snapshot of the pre-mutation layer slice — deep copy
  of weight tensors is NOT required (axon slices are replaced atomically).
- `TopologyVersion` is a `uint64` stored on `Network[T]`; exposed via the observability interface.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| DYN-1 Mode fixed at Compile()-time | `TopologyMode` field on `networkConfig`; set by `WithTopologyMode` option; checked in `Compile()`. Mid-life mode change returns `ErrImmutableMode`. |
| DYN-2 Mutations require Paused/Idle | Each mutation method calls `n.state.Require(Paused, Idle)` (atomic cell from `l2-control-impl`) at entry; returns `ErrInvalidState` on failure. |
| DYN-3 Transactional mutations | `topologyTx` captures pre-mutation layer slice; `Commit()` replaces; `Rollback()` restores on validation error. |
| DYN-4 Deterministic rebalance | `rebalance(rng seed)` re-wires axons in a fixed order (predecessor → new layer → successor) using the network's configured RNG seed. |
| DYN-5 TopologyVersion counter | `uint64` field on `Network[T]`; incremented inside `topologyTx.Commit()`; exposed via `TopologyVersion() uint64`. |
| DYN-6 Minimum topology constraint | `RemoveHiddenLayer` checks `len(hidden) > 1` before proceeding; returns `ErrMinimumTopology` if only one hidden layer remains. |

## 5. Detailed Design

### 5.1 Package Structure

```plaintext
pkg/network/
├── topology.go        # TopologyMode enum, mutation methods on Network[T], topologyTx
├── topology_test.go   # Unit tests for all mutation methods and error paths
pkg/nn/
├── options.go         # WithTopologyMode[T] option (adds ~1 line)
├── builder.go         # WithTopologyMode builder method (adds ~1 line)
pkg/utils/
└── errors.go          # New sentinel errors: ErrInvalidPosition, ErrImmutableLayer,
                       # ErrMinimumTopology, ErrEmptyLayer, ErrMutationFailed, ErrImmutableMode
```

### 5.2 TopologyMode

```go
// [REFERENCE]
type TopologyMode uint8

const (
    Immutable TopologyMode = iota // default — INV-2 enforced
    Dynamic                       // opt-in — mutation methods enabled
)
```

### 5.3 Mutation Method Signatures

```go
// [REFERENCE] — methods on *Network[T] in pkg/network/topology.go
func (n *Network[T]) AddHiddenLayer(position uint, size uint, activation activation.Type, bias bool) error
func (n *Network[T]) RemoveHiddenLayer(position uint) error
func (n *Network[T]) AddNeuron(layerIdx uint, count uint) error
func (n *Network[T]) RemoveNeuron(layerIdx uint, count uint) error
func (n *Network[T]) TopologyVersion() uint64
```

All methods return `ErrImmutableLayer` (wrapping the specific context) if called on an Immutable network.

### 5.4 Transaction Object

```go
// topologyTx captures pre-mutation state for rollback (package-private)
type topologyTx[T utils.Float] struct {
    net      *Network[T]
    snapshot []layer.Layer[T] // shallow copy of layer slice at tx start
}

func (tx *topologyTx[T]) Commit(newLayers []layer.Layer[T])
func (tx *topologyTx[T]) Rollback()
```

The snapshot is a slice of layer references (not deep copies of weights). This is safe because
mutation replaces whole layer objects; in-place weight modification does not occur during a tx.

### 5.5 Rebalance Algorithm

```text
rebalance(net, rng):
    for each adjacent pair (prev, next) in net.layers:
        if axons between prev→next are stale (count != prev.Size × next.Size):
            destroy stale axons
            create fresh axons with weights from init strategy
            preserve existing valid axons bit-identically
```

Weight init strategy is read from `net.config.WeightInit` (same as `Compile()`-time init).

### 5.6 Error Sentinels

New sentinels to add to `pkg/utils/errors.go` (or `pkg/network/errors.go`):

| Sentinel | Trigger |
| :--- | :--- |
| `ErrImmutableMode` | Mutation method called on Immutable network |
| `ErrInvalidState` | Network not in Paused/Idle state (already exists in control-impl — reuse) |
| `ErrInvalidPosition` | `position` out of range for hidden layers |
| `ErrImmutableLayer` | Attempt to mutate Input or Output layer |
| `ErrMinimumTopology` | `RemoveHiddenLayer` would leave 0 hidden layers |
| `ErrEmptyLayer` | `RemoveNeuron` would leave 0 cells in a layer |
| `ErrMutationFailed` | Validation failed after mutation; rolled back |

### 5.7 Observability Integration

`TopologyVersion()` MUST be included in the observability snapshot returned by `l2-visualization-api`
and in checkpoint metadata (`pkg/checkpoint/writer.go`). Add `TopologyVersion uint64` to the
checkpoint JSON schema.

## 6. Implementation Notes

1. **Phase A**: Add `TopologyMode` enum + `WithTopologyMode` option + gate checks (no mutations yet).
   Run full test suite — zero regressions expected.
2. **Phase B**: Implement `AddNeuron` / `RemoveNeuron` (least disruptive — no inter-layer re-wire).
3. **Phase C**: Implement `AddHiddenLayer` — inserts layer + re-wires predecessor and successor.
4. **Phase D**: Implement `RemoveHiddenLayer` — removes layer + bridge re-wire + weight re-normalization.
5. **Phase E**: Expose `TopologyVersion` via observability and checkpoint. Update `pkg/checkpoint/writer.go`.
6. Write `pkg/network/topology_test.go` alongside Phase B (TDD from Phase B onward).
7. Target ≥80% coverage on `pkg/network/topology.go` per CLAUDE.md §4.3.

## 7. Drawbacks & Alternatives

- **Alternative (pkg/topology/ new package)**: Separates concerns but requires exporting internal
  `Network[T]` fields. Keeping mutation methods on `*Network[T]` in `pkg/network/` avoids API leakage.
- **Drawback (shallow snapshot)**: Rollback restores layer references but cannot undo partial
  in-flight writes if a future mutation path modifies weights in-place. The constraint "mutation
  replaces whole layer objects" must be enforced by review.
- **Open question (optimizer state across mutations)**: After `AddNeuron`, should optimizer momentum
  buffers for unchanged weights be preserved? Likely yes — tracked as TBD in L1 §5.5.

## Canonical References

<!-- Populated at Stable promotion. Source files do not exist yet. -->

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[TOPO]` | `pkg/network/topology.go` | Mutation methods and topologyTx — created in v0.7.0 Phase A |
| `[CTRL]` | `pkg/network/state.go` | Atomic state cell (Paused/Idle guard) |
| `[LAYER]` | `pkg/layer/core.go` | Layer interface used in mutation |
| `[CELL]` | `pkg/neuron/cell/core.go` | Cell model re-wired in neuron mutations |
| `[INIT]` | `pkg/nn/options.go` | WithTopologyMode option surface |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-10 | Initial Draft — Go blueprint for l1-dynamic-topology.md realization. v0.7.0 Phase A scope. |
