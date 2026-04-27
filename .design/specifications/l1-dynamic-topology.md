# Dynamic Topology Mode

**Version:** 0.1.0
**Status:** Draft
**Layer:** concept

## Overview

Proposes a **Dynamic mode** for networks where layers and neurons can be added or removed during
training or inference. Today `l1-neural-network-architecture.md` INV-2 forbids this ("Network topology
is immutable after construction"). This spec proposes amending INV-2 to introduce two modes:

- **Immutable** (current default) — INV-2 holds as written.
- **Dynamic** — topology mutations are permitted at well-defined synchronization barriers.

> ⚠ **Conflict flag**: This spec contradicts `l1-neural-network-architecture.md` INV-2 (Stable). Resolution:
> amend the parent spec to introduce mode parameter when this spec promotes from Draft → RFC.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent (proposed amendment)
- [l1-training-control.md](l1-training-control.md) — Mutations occur at Paused-state safe points

## 1. Motivation

Real research workflows want to:

- Grow a network during training (add a hidden layer when loss plateaus — neural architecture search).
- Prune dead neurons (those whose outgoing weights have collapsed near zero).
- Add output dimensions when the task definition changes (online learning with new classes).

Forcing teardown + rebuild + retrain from scratch wastes compute and breaks continuity.

## 2. Constraints & Assumptions

- Dynamic mode is **opt-in** — the default remains Immutable to preserve current users' guarantees.
- Mutations are only permitted at **synchronization barriers** (paused state, between epochs).
  Mid-iteration mutation is an error.
- Adding capacity preserves existing weights bit-identically; only new connections are initialized fresh.
- Removing capacity invalidates affected outgoing axons; downstream weights must be re-normalized.

## 3. Core Invariants

- **DYN-1**: Mode is fixed at `Compile()`-time via `WithTopologyMode(Immutable | Dynamic)`. Switching
  mode mid-life is forbidden.
- **DYN-2**: All mutation methods (`AddLayer`, `RemoveLayer`, `AddNeuron`, `RemoveNeuron`) require the
  network to be in `Paused` (training) or `Idle` (post-train) state per `l1-training-control`.
- **DYN-3**: Mutations are **transactional** — either fully applied or fully rolled back on validation
  failure. No half-mutated network states exist.
- **DYN-4**: After any mutation, a **rebalance step** re-wires axons; the rebalance algorithm is
  deterministic given (old graph, mutation, RNG seed).

## 5. Detailed Design

### 5.1 Mutation API (proposed)

```text
nn.AddHiddenLayer(position uint, size uint, activation Type, bias bool) error
nn.RemoveHiddenLayer(position uint) error
nn.AddNeuron(layerIdx uint, count uint) error
nn.RemoveNeuron(layerIdx uint, count uint) error
```

### 5.2 Open Questions

- <!-- TBD: gradient continuity across mutation — should optimizer state for unchanged weights persist? probably yes -->
- <!-- TBD: what happens to in-flight epoch counter / loss history after mutation? -->
- <!-- TBD: serialization format extension for `Dynamic` snapshots (history of mutations) -->
- <!-- TBD: amendment mechanics for parent INV-2 — mode parameter or two-flavor concept -->

## 6. Implementation Notes

1. Phase 1: define mode parameter and gate all mutations behind `Dynamic` mode.
2. Phase 2: implement `AddNeuron` (least disruptive) first as proof-of-concept.
3. Phase 3: implement `AddHiddenLayer` (re-wires more axons but same algorithm).
4. Phase 4: removal operations (require careful weight re-normalization).

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[L1-ARCH]` | `.design/specifications/l1-neural-network-architecture.md` | Parent — INV-2 to be amended |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #6 — flagged conflict with INV-2. |
