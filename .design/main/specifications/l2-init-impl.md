# Weight Initialization Implementation

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-weight-initialization.md

## Overview

Concrete Go realization of [l1-weight-initialization.md](l1-weight-initialization.md): the
`pkg/utils/init.go` (or `pkg/init/`) sampling functions for Xavier / He / Random, RNG plumbing
using stdlib `math/rand/v2`, and the wiring point inside `Compile()` that drives them per layer.

## Related Specifications

- [l1-weight-initialization.md](l1-weight-initialization.md) — Parent — invariants WI-1..WI-4 + formulas
- [l2-nn-facade.md](l2-nn-facade.md) — `Compile()` calls the entry point defined here

## 1. Motivation

L1 pins the formulas and the RNG-seed contract. This spec decides the Go API — function signatures,
where seeded RNG lives in the `*NN[T]` instance, and the order in which layers are initialized.

## 2. Constraints & Assumptions

- Stdlib only: `math/rand/v2`, `math`.
- One `*rand.Rand` per `*NN[T]` instance, owned by the network and re-used for any further random
  sampling needs (snapshot interval jitter, dropout when it lands).
- `math/rand/v2.NewPCG` chosen as the deterministic generator — fast, reproducible across architectures.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| WI-1 (Once only) | Init is invoked from `Compile()` and gated by a `n.initialized` boolean — second call is a no-op + warn. |
| WI-2 (Seed determinism) | Same `rng_seed` + same topology ⇒ same `Rand` sequence ⇒ same weights, byte-identical when serialized. |
| WI-3 (Closed-form formulas) | Each helper has the formula in a doc comment with citation; magic numbers banned. |
| WI-4 (Default = Xavier) | If `WeightInit == ""` at `Compile()`, set to `WeightInitXavier` before dispatch. |

## 5. Detailed Design

### 5.1 Helper Signatures

```go
// [REFERENCE] In pkg/utils/init.go (or pkg/init/init.go).
package utils

import (
    "math"
    "math/rand/v2"
)

// XavierUniform samples from U[-a, a], a = sqrt(6 / (fan_in + fan_out)).
// Reference: Glorot & Bengio 2010.
func XavierUniform[T Float](rng *rand.Rand, fanIn, fanOut int) T {
    a := math.Sqrt(6.0 / float64(fanIn+fanOut))
    return T(rng.Float64()*2*a - a)
}

// HeNormal samples from N(0, σ²), σ = sqrt(2 / fan_in).
// Reference: He et al. 2015.
func HeNormal[T Float](rng *rand.Rand, fanIn int) T {
    sigma := math.Sqrt(2.0 / float64(fanIn))
    return T(rng.NormFloat64() * sigma)
}

// Uniform samples from U[-1, 1).
func Uniform[T Float](rng *rand.Rand) T {
    return T(rng.Float64()*2 - 1)
}
```

### 5.2 Compile-time Wiring (pseudo-code)

```text
Compile(cfg):
    seed := cfg.rng_seed
    if seed == 0:
        seed = uint64(time.Now().UnixNano())
        cfg.rng_seed = 0  // marker for "non-reproducible" in persisted form
    n.rng = rand.New(rand.NewPCG(seed, seed))

    for each layer L in network:
        fanIn  = L.predecessor.size
        fanOut = L.size
        for each neuron in L:
            for each axon a:
                a.weight = sample(cfg.WeightInit, n.rng, fanIn, fanOut)
            neuron.bias = 0
    n.initialized = true

sample(method, rng, fanIn, fanOut):
    switch method:
        case xavier: return XavierUniform(rng, fanIn, fanOut)
        case he:     return HeNormal(rng, fanIn)
        case random: return Uniform(rng)
        default:     return error wrapping ErrUserConfig
```

### 5.3 Open Questions

- <!-- TBD: per-layer WeightInit override (l1 §5.4 reserved) — extend Layer config struct now or later -->
- <!-- TBD: WeightInitAuto symbol that selects per-activation — needs activation→strategy table -->

## 6. Implementation Notes

1. Live in `pkg/utils/init.go` to share with downstream features that need RNG sampling.
2. `*rand.Rand` is **not** safe for concurrent use; the training loop is single-goroutine so this is fine.
3. RNG is persisted in snapshots (per `l1-checkpointing.md` CHK-2) — the underlying PCG state must be
   serializable. `math/rand/v2.PCG` exposes a `MarshalBinary`/`UnmarshalBinary` API.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[INIT]` | `pkg/utils/init.go` (new) | Sampling helpers + RNG plumbing |
| `[FACADE-RES]` | `.design/main/specifications/l2-nn-facade.md#56-weight-initialization-methods` | Reserved symbols this realizes |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-28 | Initial Draft — concrete Go realization of l1-weight-initialization RFC. |
| 0.2.0 | 2026-04-28 | Status promoted Draft → RFC after parent l1-weight-initialization reached Stable. Sampling helpers + Compile() wiring ready for review. |
| 1.0.0 | 2026-04-30 | Promoted RFC → Stable. Validated by Phase-1 implementation: pkg/utils/init.go ships zero-allocation hot path (15-42 ns/op), reproducibility test, distribution mean/variance within 5% over n=20000, 100% test coverage, race-detector clean. |
