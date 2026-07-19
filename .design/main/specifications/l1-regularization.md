# Regularization

**Version:** 1.0.0
**Status:** Stable
**Layer:** concept

## Overview

Defines the abstract contract for techniques that reduce model overfitting by constraining weight
magnitudes or randomly deactivating neurons during training. Regularization is applied at two
points in the training loop: (1) a weight penalty added to the loss before gradient computation
(L1/L2 penalty), and (2) a mask applied to layer activations during the forward pass (Dropout).
This spec is technology-agnostic; `l2-regularization-impl.md` realizes it in Go.

## Related Specifications

- [l1-training-semantics.md](l1-training-semantics.md) — Training loop that applies regularization
- [l1-math-functions.md](l1-math-functions.md) — Loss functions augmented by penalty term
- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Network structure defining weight tensors
- [l1-weight-initialization.md](l1-weight-initialization.md) — RNG seed contract reused for Dropout

## 1. Motivation

GoNN's multi-hidden networks (v0.6 Phase 5) can now represent complex functions. Without
regularization, deeper networks memorize training data instead of generalizing — the classic
overfitting problem. Known symptoms: test loss rises while training loss falls; deep ReLU chains
accumulate dead neurons.

Three canonical techniques cover the majority of practical cases:

- **L2 weight decay** — most common default; penalizes large weights.
- **L1 weight sparsity** — promotes near-zero weights (feature selection).
- **Dropout** — randomly deactivates neurons during training (ensemble approximation).

## 2. Constraints & Assumptions

- Regularization is **opt-in** and **per-network** — default is no regularization (zero penalty,
  identity mask).
- L1/L2 regularizers are applied to **weight tensors only** — bias weights are excluded
  (standard practice; biases are not regularized to allow fitting the data mean).
- Dropout is applied **during training only**. During inference (`Query()`), Dropout is a no-op
  and the network uses full activations scaled by the retention probability (inverted dropout).
- Regularization MUST NOT alter the gradient computation logic — it augments the effective loss,
  not the backprop algorithm.
- Multiple regularizers may be composed: `Compose(L2(λ), Dropout(p))`.

## 3. Core Invariants

- **REG-1**: Every regularizer implements `Penalty(weights []T) T` — returns the scalar penalty
  to be added to the scalar loss before gradient computation.
- **REG-2**: Every regularizer implements `ApplyMask(activations []T, training bool) []T` —
  for non-Dropout regularizers, this is an identity function.
- **REG-3**: When `training == false`, `ApplyMask` MUST return activations unmodified and MUST
  NOT consume RNG state. Inference is deterministic.
- **REG-4**: The `nil` regularizer (default) returns `0` from `Penalty` and the unmodified slice
  from `ApplyMask` — zero performance cost beyond a nil check.
- **REG-5**: Dropout retains each activation with probability `p ∈ (0, 1]` and scales retained
  values by `1/p` (inverted dropout). `p == 1.0` is equivalent to no Dropout.
- **REG-6**: A composed regularizer applies `Penalty` additively and `ApplyMask` sequentially
  left-to-right. The combined penalty is the sum of individual penalties.

## 5. Detailed Design

### 5.1 Algorithm Reference

| Name | Penalty formula | ApplyMask (training) |
| :--- | :--- | :--- |
| L2 (Ridge) | λ × Σwᵢ² | identity |
| L1 (Lasso) | λ × Σ\|wᵢ\| | identity |
| Dropout(p) | 0 | retain with prob p; multiply retained by 1/p; zero others |
| Compose(…) | Σ penalties | sequential ApplyMask |

### 5.2 Application in Training Loop

```text
// inside each training iteration:
effectiveLoss ← loss(predicted, target) + reg.Penalty(allWeights)
gradients     ← backprop(effectiveLoss)
hiddenActs    ← reg.ApplyMask(hiddenActs, training=true)
```

### 5.3 Inference Behaviour (REG-3 elaboration)

During `Query()` the training loop invokes `ApplyMask(acts, false)`. All implementations
MUST return `acts` unchanged. This contract allows the same regularizer instance to be used
for both training and inference without conditional branching in the caller.

## 6. Implementation Notes

1. Implement L2 first (no RNG dependency, validates Penalty path end-to-end).
2. L1 follows L2 (same interface, different formula).
3. Dropout requires seeded RNG — coordinate with the RNG seed contract in
   `l1-weight-initialization.md`.
4. Compose last — validates multi-regularizer interaction.
5. Overfitting validation: Dropout should visibly reduce test loss on the Iris example (E05)
   when the training set is artificially reduced.

## 7. Drawbacks & Alternatives

- **Batch Normalization** is a more powerful regularization technique but requires tracking
  running mean/variance and fundamentally changes the forward pass — deferred to a future
  `l1-batch-norm.md` spec.
- A weight-decay term injected directly into the training loop (no interface) suffices for L2
  only, but blocks Dropout composition and meta-learning parameter access.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[TRAIN-L1]` | `.design/main/specifications/l1-training-semantics.md` | Training loop invariants that regularization must not violate |
| `[MATH-L1]` | `.design/main/specifications/l1-math-functions.md` | Loss function contract that penalty term augments |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-05-07 | Initial — regularization contract for Phase 6 (Spark 2). Trust Mode Stable. |
