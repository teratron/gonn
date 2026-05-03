# Multi-Hidden Topology Implementation

**Version:** 0.1.0
**Status:** Draft
**Layer:** implementation
**Implements:** l1-neural-network-architecture.md

## Overview

Concrete realization of the multi-hidden-layer topology already permitted by
[l1-neural-network-architecture.md](l1-neural-network-architecture.md) v2.0.0
but rejected by the v0.1 `pkg/nn.compile()` validation gate. This spec lifts
the gate, generalizes `pkg/network.Network[T]`'s single-bundle Hidden field
to a chain of bundles, and re-routes forward / backward / weight-update so
gradients propagate through every intermediate layer rather than only one.

The unlock matters because eight catalog examples
([l2-usage-examples.md](l2-usage-examples.md) §5.2 — E03, E04, E05, E06,
E07, E08, E10, E13) describe topologies with two or more hidden layers and
are explicitly deferred to v0.2 in [tasks/phase-4.md](../tasks/phase-4.md).
Lifting the constraint here unblocks seven of those entries directly
(E10 still waits on `AndTrain`, E06 on a dataset-loader spec).

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent. INV-2 already permits arbitrary hidden depth.
- [l2-nn-facade.md](l2-nn-facade.md) — Public facade whose `compile()` currently rejects `len(HiddenLayers) > 1`. v2.1.0 minor bump consumes this spec.
- [l2-network-graph.md](l2-network-graph.md) — Owns `Network[T]`. v1.2.0 minor bump generalizes the Hidden bundle to a slice.
- [l2-neuron-model.md](l2-neuron-model.md) — Cell / axon types reused unchanged.
- [l2-usage-examples.md](l2-usage-examples.md) — Eight v0.2 catalog entries unblocked by this spec.

## 1. Motivation

`pkg/nn.compile()` (Phase 2 outcome, 2026-04-30) shipped with a hard guard:

```text
if len(cfg.HiddenLayers) > 1 → ErrUserConfig "multi-hidden networks not
supported in v0.1 (got N hidden layers; planned for v0.2)"
```

The guard was deliberate — Phase 2 stopped at the smallest topology that
exercises every public API element so the facade could stabilize without
also debugging the propagation chain. Phase 4 then partitioned the example
catalog into v0.1 (single-hidden) and v0.2 (multi-hidden) groups; eight
entries plus several catalog API surfaces (`Sequential`, `DeepNetwork`,
`PresetMNIST`, `PresetRegression`, `Verify`) wait for this spec.

Removing the guard requires three coordinated changes:

1. **Validation lift** in `pkg/nn.compile()` — accept any positive
   `len(cfg.HiddenLayers)` and propagate the chain into `Network[T]`.
2. **Topology storage** in `pkg/network.Network[T]` — replace the single
   `Hidden` bundle with a slice of bundles preserving order.
3. **Propagation routing** — Forward, Backward, and UpdateWeights iterate
   the chain, threading activations / pre-activations / gradients through
   every layer instead of one.

## 2. Constraints & Assumptions

- Stdlib only per C29; no new dependencies.
- v0.1 single-hidden behaviour is preserved exactly when `len(HiddenLayers) == 1`
  — the new chain reduces to the existing path; no public API breakage.
- Activation per hidden layer is read from `cfg.HiddenLayers[i].Activation`
  (already present in `Config[T]`); a single uniform activation across the
  chain is no longer assumed.
- Bias per hidden layer is per-`HiddenLayerSpec[T].Bias` (already present);
  mixed-bias chains are supported.
- Soft-warning chain in `compile()` (`emitSoftWarnings`) is extended to the
  new "deep stack with WeightInitRandom" check from
  [l1-neural-network-architecture.md](l1-neural-network-architecture.md) §6 —
  triggered when `len(cfg.HiddenLayers) > 5 && WeightInit == Random`.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| INV-1 (Generic Float) | Type parameter `T utils.Float` flows through every new API; no concrete float type leaks. |
| INV-2 (Immutable topology) | `Network[T]` exposes `Hiddens` as a fixed-size slice set once at `Build()` time; no public method appends or removes after Compile. |
| INV-3 (Forward pass) | New `propagateForward` iterates `n.Hiddens` left-to-right, feeding each layer's outputs as the next layer's inputs. Activations are stored in `n.preactHiddens[i]` for backprop. |
| INV-4 (Backward pass) | New `propagateBackward` iterates `n.Hiddens` right-to-left, threading `δ = (W^T · δ_next) ⊙ σ'(z_i)` through every chain step. UpdateWeights then updates each layer's axons using the cached preactivations. |
| INV-7 (Layer composition) | Hidden chain reuses `*cell.Hidden[T]` and `axon.Bundle[T]` unchanged; only the bundle slice is new. |

## 5. Detailed Design

### 5.1 Network[T] storage shape (delta vs. v1.1.0)

```go
// [REFERENCE] Replacement for pkg/network.Network[T].Hidden field.
// One bundle per hidden layer, in compile-time order. Length is
// determined by SetLayers and never mutated after Build.
type Network[T utils.Float] struct {
    LearningRate T
    Input        bundle[T, *cell.Input[T]]
    Hiddens      []bundle[T, *cell.Hidden[T]]   // was: Hidden bundle[...]
    Output       bundle[T, *cell.Output[T]]

    hiddenBiases   []*cell.Bias[T]              // was: hiddenBias *cell.Bias[T]
    outputBias     *cell.Bias[T]
    hiddenActs     []activation.Type            // was: hiddenAct activation.Type
    outputAct      activation.Type
    lossMode       loss.Type

    preactHiddens  [][]T                        // was: preactHidden []T
    preactOutput   []T
}
```

Field count grows by zero — three pluralised fields replace three singular
ones. The JSON tags on `Hiddens` shift the persistence wire format from
`"hidden"` (object) to `"hiddens"` (array of objects); see §5.4.

### 5.2 SetLayers signature (delta)

```go
// [REFERENCE] pkg/network.Network[T].SetLayers — variadic hidden chain.
func (n *Network[T]) SetLayers(
    in *layer.Input[T],
    hiddens []*layer.Dense[T],
    out *layer.Output[T],
) error
```

Hiddens is a slice rather than a single value. Validation rejects empty
slice (`len(hiddens) == 0`) and any nil element. Single-hidden callers
pass `[]*layer.Dense[T]{h}` — a one-line migration in `pkg/nn.compile()`.

### 5.3 Build wiring chain

```text
for i, h := range n.Hiddens:
    src = if i == 0 then n.Input else n.Hiddens[i-1]
    for each cell c in h:
        c.Axons = []
        for each src_cell s in src:
            c.Axons.append(axon.New(s, c))
        if n.hiddenBiases[i] != nil:
            c.Axons.append(axon.New(n.hiddenBiases[i], c))

for each cell o in n.Output:
    o.Axons = []
    for each src_cell s in n.Hiddens[last]:
        o.Axons.append(axon.New(s, o))
    if n.outputBias != nil:
        o.Axons.append(axon.New(n.outputBias, o))
```

Bias cells are stored per-hidden-layer; layers with `Bias == false`
contribute `nil` so the lookup remains positional.

### 5.4 Persistence wire format extension (PERS-1 minor bump)

`persistence.ConfigDoc[T]` already serialises `HiddenLayers []HiddenLayerDoc`
as an array (Phase 3 Track A); no schema change there. `WeightsDoc[T]`
generalises the existing two-layer convention to N+1 entries:

```json
{
  "schema_version": "1.1.0",
  "config_hash": "sha256:...",
  "layers": [
    {"name": "hidden_0", "weights": [[...]], "biases": [...]},
    {"name": "hidden_1", "weights": [[...]], "biases": [...]},
    {"name": "output",   "weights": [[...]], "biases": [...]}
  ]
}
```

Schema version bumps `1.0.0 → 1.1.0` (minor, forward-compatible). v0.1 readers
loading a v0.2 file see the new entries and apply the existing forward-compat
rule from [l1-network-persistence.md](l1-network-persistence.md) §3 PERS-1
(minor mismatch → warning + best-effort load — load fails because cell counts
will not match, but the failure is `ErrIntegrity` rather than a parse crash).
v0.2 readers loading a v0.1 file see one hidden layer and rebuild correctly.

### 5.5 compile() validation delta

```go
// [REFERENCE] Replacement for the v0.1 multi-hidden guard in pkg/nn/compile.go.
//   Removed:
//     if len(cfg.HiddenLayers) > 1 { return ErrUserConfig "...not supported in v0.1..." }
//   Added (validation pass):
//     for i, h := range cfg.HiddenLayers {
//         if h.Size == 0 → ErrUserConfig "hidden layer %d has size 0"
//         if !isKnownActivation(h.Activation) → ErrUserConfig "..."
//     }
//   Added (build pass):
//     hiddens := make([]*layer.Dense[T], len(cfg.HiddenLayers))
//     for i, hSpec := range cfg.HiddenLayers {
//         hiddens[i] = layer.NewDense[T](int(hSpec.Size), hSpec.Activation, hSpec.Bias)
//     }
//     n.SetLayers(in, hiddens, out)
```

The existing per-hidden Size / Activation validation loop in `validate()` is
already in place (Phase 2 wrote it); only the gate immediately above it is
removed.

### 5.6 Forward / Backward pseudocode

```text
Forward(input):
    n.Input.SetValues(input)
    prev = n.Input
    for i, h := range n.Hiddens:
        for each cell c in h:
            sum = Σ(a.Weight × a.Cell.Value() for a in c.Axons)
            n.preactHiddens[i][c.idx] = sum
            c.SetValue(activation.Activation(sum, n.hiddenActs[i]))
        prev = h
    for each cell o in n.Output:
        sum = Σ(a.Weight × a.Cell.Value() for a in o.Axons)
        n.preactOutput[o.idx] = sum
        o.SetValue(activation.Activation(sum, n.outputAct))

Backward(target):
    // Output deltas
    for each cell o in n.Output:
        miss = target[o.idx] - o.Value()
        o.SetMiss(miss × activation.Derivative(n.preactOutput[o.idx], n.outputAct))

    // Hidden deltas — right-to-left
    for i := len(n.Hiddens)-1; i >= 0; i--:
        next_layer_cells = (i == len(n.Hiddens)-1) ? n.Output : n.Hiddens[i+1]
        for each cell c in n.Hiddens[i]:
            agg = Σ(a.Weight × a.OutgoingCell.Miss() for a in next_axons_pointing_to(c))
            c.SetMiss(agg × activation.Derivative(n.preactHiddens[i][c.idx], n.hiddenActs[i]))

UpdateWeights(rate):
    for each cell o in n.Output:
        for each a in o.Axons:
            a.Weight += rate × o.Miss() × a.Cell.Value()
    for i, h := range n.Hiddens:
        for each cell c in h:
            for each a in c.Axons:
                a.Weight += rate × c.Miss() × a.Cell.Value()
```

The "next_axons_pointing_to(c)" lookup avoids a reverse adjacency map by
walking the next layer's cells once per backward pass and accumulating into
each source cell — the layer chain is short (typically ≤ 5 in practice) so
the O(N²) cost stays within PERF-2 budgets.

### 5.7 Open Questions

- <!-- TBD: should `Sequential(count, size, activation)` and `DeepNetwork(start, depth, activation)` validate that the resulting chain length stays under a soft cap (e.g. 64)? Spec says no, but l1-performance-contract.md PERF-3 worker pool budgets benefit from a bound. Default to no cap for v0.2; reconsider after benchmarks. -->
- <!-- TBD: how does WeightInit propagate per-layer? Phase 2 used a single `cfg.WeightInit` applied uniformly. Multi-hidden may want per-layer init (Xavier for tanh hidden, He for ReLU hidden). Out of scope for the v0.2 lift — defer to a follow-up minor when call sites surface the need. -->

## 6. Implementation Notes

1. **Order of land**: pkg/network first (storage + Build wiring + Forward/Backward/UpdateWeights), then pkg/nn (compile() validation + SetLayers call), then examples (E03/E04/E05/E07/E08/E13 in `examples/`).
2. **Backwards compatibility test**: `pkg/nn.PresetXOR` keeps the single-hidden topology — every v0.1 example must continue to converge after the lift. Phase 4 v0.1 smoke tests ARE the regression suite for this guarantee.
3. **Persistence migration**: bump `persistence.SchemaVersion` to `"1.1.0"` and add a regression test loading the v0.1 fixtures generated by `examples/persistence/`.
4. **Coverage**: `pkg/network` regression is required at the propagation level — table-driven tests with `[][]T` weights for two-hidden and three-hidden chains, comparing against hand-computed reference vectors (analogous to E01 cpu kernel tests in `pkg/compute/cpu/cpu_test.go`).

## 7. Drawbacks & Alternatives

- **Drawback (storage churn)**: shifting `Hidden` → `Hiddens []bundle` is a minor breaking change to anything that imported `pkg/network.Network[T]` directly. Mitigation: only `pkg/nn` reaches into the bundle; the public surface is the facade, which masks the change.
- **Alternative (fixed-N union)**: keep `Hidden`, `Hidden2`, `Hidden3` as discrete fields up to some cap. Rejected — does not scale, complicates cell iteration, and forces every consumer to know the cap.
- **Alternative (linked list of layers)**: store hidden cells via `next`/`prev` pointers on each layer. Rejected — slice-of-bundles preserves Go cache locality and simplifies iteration order; pointer chasing buys nothing semantically.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[NETWORK]` | `pkg/network/network.go` | Hosts the storage and propagation changes |
| `[FACADE]` | `pkg/nn/compile.go` | Validation gate lift + variadic `SetLayers` call |
| `[PERSIST]` | `pkg/persistence/weights.go` | Multi-layer wire format consumer |
| `[EXAMPLES]` | `examples/E03..E13` | Eight catalog entries unblocked by this spec |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-03 | Initial Draft — concrete v0.2 plan for lifting the `len(HiddenLayers) > 1` rejection in `pkg/nn.compile()`. Documents the storage shape change in `Network[T]`, the propagation chain across the new hidden slice, and the matching weights schema bump (1.0.0 → 1.1.0). Drives Phase 5 (v0.2) decomposition. |
