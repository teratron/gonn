---
phase: 6
name: Feature Expansion + v0.6 Release
status: Todo
subsystem: pkg/optimizer, pkg/regularizer, pkg/nn, pkg/neuron/axon, root docs
requires:
  - phase-5 (multi-hidden topology complete, all tracks green)
provides: []
key_files: []
patterns_established: []
duration_minutes:
---

# Phase 6 — Feature Expansion + v0.6 Release

**Status:** Active
**Decomposed:** 2026-05-07
**Tasks:** 15 feature + 2 validation + 2 gate = 19 total
**Specs:** l1-optimizer-strategies v1.0.0, l2-optimizer-impl v1.0.0,
           l1-regularization v1.0.0, l2-regularization-impl v1.0.0,
           l1-release-policy v1.0.0
**Track order:** A ∥ B ∥ C (parallel); T-6T01 after A; T-6T02 after B; Gate T-6Z after all tracks.

## Track A — Optimizer Strategies

*Goal: `pkg/optimizer/` package with SGD/Adam/RMSProp/Momentum + pkg/nn integration.*
*Source: [l2-optimizer-impl.md](../specifications/l2-optimizer-impl.md)*

- [x] **T-6A01** — Create `pkg/optimizer/optimizer.go`: define `Optimizer[T Float]` interface
  (`Step`, `Reset`, `LearningRate`, `SaveState`, `LoadState`) + `DefaultOptimizer[T](lr T)`.
  Add `go.mod` entry for the new package.

- [x] **T-6A02** — Implement `pkg/optimizer/sgd.go`: `SGD[T]` — stateless, zero-LR guard,
  OPT-3 compliance. Target: 0 allocs/op in hot path.

- [x] **T-6A03** — Implement `pkg/optimizer/adam.go`: `Adam[T]` — first + second moment slices
  lazily preallocated on first `Step`; step counter `t`; JSON serialization for SaveState/LoadState.
  Default hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8.

- [x] **T-6A04** — Implement `pkg/optimizer/sgd_momentum.go` (`SGDMomentum[T]`, γ=0.9 default)
  and `pkg/optimizer/rmsprop.go` (`RMSProp[T]`, α=0.99, ε=1e-8 default).

- [x] **T-6A05** — Wire `WithOptimizer(opt Optimizer[T])` into `pkg/nn/options.go` and resolve
  in `pkg/nn/compile.go`: if `cfg.Optimizer == nil`, fall back to `optimizer.DefaultOptimizer[T](cfg.LearningRate)`.

- [x] **T-6A06** — Update `pkg/nn/train.go`: replace the inline `rate × delta` weight-update with
  `opt.Step(weights, deltas)`. Verify single-hidden XOR example (E01) is bit-identical to pre-change
  output when `WithOptimizer(NewSGD(lr))` is used.

## Track B — Regularization

*Goal: `pkg/regularizer/` package with L1/L2/Dropout/Compose + pkg/nn integration + axon WeightInit fix.*
*Source: [l2-regularization-impl.md](../specifications/l2-regularization-impl.md)*

- [x] **T-6B01** — Create `pkg/regularizer/regularizer.go`: define `Regularizer[T Float]`
  interface (`Penalty`, `ApplyMask`) + nil-guard helper (`Apply(reg, acts, training)`).

- [x] **T-6B02** — Implement `pkg/regularizer/l2.go` (`L2[T]`, penalty = λ × Σwᵢ²) and
  `pkg/regularizer/l1.go` (`L1[T]`, penalty = λ × Σ|wᵢ|). Both `ApplyMask` are identity.

- [x] **T-6B03** — Implement `pkg/regularizer/dropout.go`: `Dropout[T]`, Bernoulli mask with
  probability `p`, inverted scaling (`×1/p` on retained). RNG seeded from `pkg/utils/init.go`
  PCG source. `ApplyMask(_, false)` is a strict no-op (REG-3).

- [x] **T-6B04** — Implement `pkg/regularizer/compose.go`: `Compose[T]` combinator — additive
  `Penalty` + sequential `ApplyMask` left-to-right over a `[]Regularizer[T]` slice.

- [x] **T-6B05** — Wire `WithRegularizer(reg Regularizer[T])` into `pkg/nn/options.go`. In
  `pkg/nn/train.go` apply penalty before backprop and `ApplyMask(acts, true)` after forward pass.
  In `pkg/nn/nn.go` Query path: `ApplyMask(acts, false)`.
  **Note**: apply to `train.go` after T-6A06 is committed to avoid line-level conflicts in the
  iteration loop — both tasks edit the same file at different call sites.

- [x] **T-6B06** — Fix known debt: `pkg/neuron/axon/axon.go` `New[T]()` MUST apply the
  configured `WeightInit` strategy (Xavier / He / Random) when initializing axon weights in
  `Build()`. Current behaviour: always uses `U[-0.5, 0.5]`. Fix must be backward-compatible
  (existing tests with no explicit WeightInit must still pass; Xavier for SIGMOID is the default).

## Track C — v0.6 Release Preparation

*Goal: public API audit, CHANGELOG, README, v0.6.0 tag.*
*Source: [l1-release-policy.md](../specifications/l1-release-policy.md)*

- [x] **T-6C01** — Audit public API surface against REL-2: enumerate all exported symbols in
  `pkg/nn`, `pkg/activation`, `pkg/loss`. Document any symbols that are unintentionally exported
  (unexported candidates). Fix any godoc comments that are missing on exported symbols.

- [x] **T-6C02** — Create `CHANGELOG.md` at repo root with a `## [0.6.0] — 2026-05-08` section.
  Include: optimizer + regularizer packages, multi-hidden topology, persistence schema 1.1.0,
  six new examples (E03/E04/E05/E07/E08/E13), WeightInit debt fix, deferred items (E06, E10).

- [x] **T-6C03** — Update root `README.md`: add v0.6 section documenting optimizer/regularizer
  options, multi-hidden example snippets, weight-init table, and the new example catalog table
  (13 active + 2 deferred).

## Validation Tasks

- [x] **T-6T01** — Optimizer validation (after Track A complete):
  - `go test -race ./pkg/optimizer/...` — all rows of §5.4 test matrix green.
  - `go test -bench=. -benchmem ./pkg/optimizer/...` — `BenchmarkSGDStep` and `BenchmarkAdamStep` at 0 allocs/op.
  - `go test -race ./pkg/nn/...` — XOR (E01) bit-identical output with `NewSGD(lr)`.
  - Coverage `pkg/optimizer/`: 98.9%.

- [x] **T-6T02** — Regularization validation (after Track B complete):
  - `go test -race ./pkg/regularizer/...` — all rows green; coverage 80.0%.
  - Integration: `TestRegularizerConvergence` + `TestInferenceNoDropout` in `pkg/nn/phase6_test.go`.
  - `TestWeightInitRanges` in `pkg/nn/phase6_test.go` — Xavier/He/Random weight ranges verified.
  - `TestOptimizerIntegration` — all 4 optimizer types converge on XOR.

## Gate

- [x] **T-6Z01** — Release gate (REL-5): `go build ./...` clean; `go test -race ./...` all green;
  every non-example package ≥ 80% line coverage; `CHANGELOG.md` v0.6.0 entry present.
  Coverage: optimizer 98.9%, regularizer 80.0%, network 96.3%, nn 86.4%. Pre-existing
  TestPauseResumeCycle timing flake documented in CHANGELOG.md Known Issues.

- [x] **T-6Z02** — Tag `v0.6.0`: annotated git tag created on develop branch.
  Push and GitHub Release pending explicit user action.
