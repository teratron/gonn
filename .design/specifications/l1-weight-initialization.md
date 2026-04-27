# Weight Initialization

**Version:** 0.1.0
**Status:** Draft
**Layer:** concept

## Overview

Defines the formulas, default selection, and contract for the weight-initialization strategies
reserved as symbols in `l2-nn-facade.md` §5.6 (`xavier`, `he`, `random`). Establishes the random
source contract so initialization is reproducible across runs given a pinned seed.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent (INV-2 immutable topology means weights are set once)
- [l2-nn-facade.md](l2-nn-facade.md) — Reserved symbols `WeightInitXavier/He/Random`
- [l1-training-semantics.md](l1-training-semantics.md) — Determinism (TRN-6) depends on pinned-seed initialization

## 1. Motivation

Weight initialization is unglamorous but high-impact. A poor initialization makes a network either
fail to learn (gradient vanishing) or diverge (gradient explosion). The community converged on two
strategies — **Xavier** (Glorot) for sigmoid/tanh, **He** for ReLU family — plus a uniform-random
fallback. Pinning the formulas in a spec prevents subtle drift and makes results reproducible across
library versions.

## 2. Constraints & Assumptions

- Random number generation uses stdlib `math/rand/v2` (Go 1.22+).
- One global `*rand.Rand` per `*NN[T]` instance, seeded from `Config[T].rng_seed` (per
  `l1-network-persistence.md`).
- Initialization happens **once** at `Compile()` and is part of the persisted config (`rng_seed`)
  — re-running with the same seed produces bit-identical initial weights.
- Bias terms initialize to **zero** unconditionally (community standard, no need for parameterization).

## 3. Core Invariants

- **WI-1**: Every weight in the network is set by exactly one initialization call. Re-initialization
  is forbidden — it would violate INV-2 of the parent.
- **WI-2**: A pinned RNG seed (`rng_seed != 0`) makes initialization **deterministic**. Seed `0`
  means "auto-pick from time" and is **not** reproducible (recorded in config as `null` on save).
- **WI-3**: Each initialization strategy has a closed-form formula documented here. No
  framework-specific magic numbers ("2.0/n_in" type expressions are forbidden in code without a
  comment pointing to this spec).
- **WI-4**: The default strategy is **Xavier** (matches `l2-nn-facade.md` v2.0 default). Selection
  by activation type is **opt-in** via a future `WeightInitAuto` symbol that reads the layer's
  activation and picks Xavier or He accordingly — reserved for v2 of this spec.

## 5. Detailed Design

### 5.1 Strategies

For a fully-connected layer with `fan_in` incoming connections per neuron and `fan_out` outgoing,
weight `w` is sampled from a distribution:

| Strategy | Distribution | Formula | When to use |
| :--- | :--- | :--- | :--- |
| `xavier` (Glorot) | Uniform `[-a, a]` | `a = sqrt(6 / (fan_in + fan_out))` | Sigmoid, TanH layers |
| `he` (Kaiming) | Normal `N(0, σ²)` | `σ = sqrt(2 / fan_in)` | ReLU, LeakyReLU layers |
| `random` | Uniform `[-1, 1]` | (constant range) | Fallback / debugging |

References: Glorot & Bengio 2010; He et al. 2015. Citation links MAY be added when the spec promotes
to RFC.

### 5.2 Selection Rule (per call site)

`Compile()` reads `Config[T].WeightInit` (one of the three symbols). The chosen strategy applies to
**every** layer uniformly. Per-layer strategies are an anticipated extension noted in §5.4.

### 5.3 RNG Source Contract

```text
At Compile():
    if config.rng_seed == 0:
        seed = uint64(time.Now().UnixNano())
        config.rng_seed = nil   # marks "non-reproducible" in persisted config
    else:
        seed = config.rng_seed
    rng = rand.New(rand.NewPCG(seed, seed))
    for each layer L:
        for each weight w in L:
            w = sample(strategy, fan_in, fan_out, rng)
        for each bias b in L:
            b = 0
```

### 5.4 Open Questions

- <!-- TBD: WeightInitAuto symbol — pick per-layer based on activation; needs activation→strategy table -->
- <!-- TBD: per-layer override — `Dense(size, act, bias).WithInit(method)` — useful for mixed networks -->
- <!-- TBD: orthogonal initialization (used for RNNs); reserved when recurrent layers land -->

## 6. Implementation Notes

1. Define `pkg/utils/init.go` with `XavierUniform`, `HeNormal`, `Uniform` sampling helpers.
2. Wire `Compile()` to invoke the helper based on `Config[T].WeightInit`.
3. Persist `rng_seed` and the initialization strategy name in `config.json` (per
   `l1-network-persistence.md`) — necessary for reproducible re-runs.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[INIT]` | `pkg/utils/init.go` (new) | Sampling helpers |
| `[FACADE-RES]` | `.design/specifications/l2-nn-facade.md#56-weight-initialization-methods` | Reserved symbols this spec realizes |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft — closes l2-nn-facade §5.6 reservation. |
