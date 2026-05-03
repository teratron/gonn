---
phase: 5
name: "Multi-Hidden Topology (v0.2)"
status: Active
subsystem: "pkg/network, pkg/nn, pkg/persistence, examples/"
requires:
  - "phase-1: pkg/utils, pkg/neuron, pkg/layer, pkg/network"
  - "phase-2: pkg/nn public facade"
  - "phase-3: pkg/persistence (weights schema baseline)"
  - "phase-4: examples/ smoke-test pattern"
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 5 Tasks — Multi-Hidden Topology (v0.2)

**Phase:** 5
**Status:** `Active`
**Strategic Goal:** Lift the v0.1 single-hidden constraint baked into `pkg/nn.compile()` per [l2-multihidden-impl.md](../specifications/l2-multihidden-impl.md) Stable v1.0.0. Deliverables span four packages — `pkg/network` (storage + propagation chain), `pkg/nn` (compile() gate removal + variadic SetLayers wiring), `pkg/persistence` (weights schema 1.0.0 → 1.1.0), and six new / refactored examples that consume multi-hidden topologies. v0.1 single-hidden behaviour is preserved bit-for-bit when `len(HiddenLayers) == 1` so the Phase 4 smoke-test suite is the regression baseline.

## Track A — Network Storage & Propagation (`pkg/network`)

**Spec:** [l2-multihidden-impl.md](../specifications/l2-multihidden-impl.md) §5.1, §5.2, §5.3, §5.6
**L1 Parent:** l1-neural-network-architecture.md Stable v2.0.0
**Depends on:** Phase 1 (pkg/network baseline). Hot path of the phase — every other track waits on Track A.

- [x] **T-5A01** — `pkg/network/network.go`: replace `Hidden bundle[...]` with `Hiddens []bundle[...]`; `hiddenBias *cell.Bias[T]` with `hiddenBiases []*cell.Bias[T]`; `hiddenAct activation.Type` with `hiddenActs []activation.Type`; `preactHidden []T` with `preactHiddens [][]T`. Update zero-value in `New[T]()` and JSON / XML struct tags (`hiddens` array of objects). Single-hidden callers continue to compile by wrapping the value in a slice. **Changes:** Network[T] storage generalised to slice form; JSON tag flipped `hidden` → `hiddens`; downstream callers in `pkg/nn` and `examples/persistence` migrated to `Hiddens[0]` access.
- [x] **T-5A02** — `pkg/network/network.go` `SetLayers` signature: change second parameter from `*layer.Dense[T]` to `[]*layer.Dense[T]`. Validate non-empty + every entry non-nil + every Size > 0. Mirror `Hiddens` slice population, `hiddenBiases[i] = hidden.BiasCell()` per layer, `hiddenActs[i] = hidden.Activation`, `preactHiddens[i] = make([]T, hidden.Size)`. Returns ErrUserConfig on shape violations per existing pattern. **Changes:** SetLayers accepts `[]*layer.Dense[T]`; per-entry nil + Size > 0 validation; positional metadata slices allocated fresh per call.
- [x] **T-5A03** — `pkg/network/network.go` `Build()`: chain wiring left-to-right per spec §5.3 — `Hiddens[0]` axons sourced from `Input`, `Hiddens[i]` axons sourced from `Hiddens[i-1]` for i ≥ 1, `Output` axons sourced from `Hiddens[len-1]`. Per-layer bias hookup uses positional `hiddenBiases[i]` lookup (nil-safe). **Changes:** Build chains Input → Hiddens[0..n-1] → Output left-to-right; bias hookup positional via `hiddenBiases[i]`.
- [x] **T-5A04** — `pkg/network/propagation.go` `forward()`: iterate `n.Hiddens` left-to-right, store each cell's pre-activation sum in `n.preactHiddens[i][cellIdx]`, set the post-activation value via `activation.Activation(sum, n.hiddenActs[i])`. Output layer logic unchanged after the chain. **Changes:** CalculateValues iterates Hiddens left-to-right; per-layer pre-activation stored in `preactHiddens[i][cellIdx]`.
- [x] **T-5A05** — `pkg/network/propagation.go` `backward()`: iterate `n.Hiddens` right-to-left per §5.6. Each cell's miss aggregates `Σ(a.Weight × a.OutgoingCell.Miss())` from the next layer (Output for last hidden, Hiddens[i+1] otherwise) and multiplies by `activation.Derivative(n.preactHiddens[i][cellIdx], n.hiddenActs[i])`. **Changes:** CalculateMisses promotes output residual to δ, walks Hiddens right-to-left, type-filters axons to `*cell.Hidden[T]` to skip biases.
- [x] **T-5A06** — `pkg/network/propagation.go` `updateWeights()`: per-layer iteration over `n.Hiddens` plus output layer; `axon.Weight += rate × cell.Miss() × axon.Cell.Value()` per axon. Preserves the v0.1 single-layer arithmetic when len == 1 (bit-identical via `T(0)`-init pre-allocs). **Changes:** CalculateWeights now passes plain rate (derivative folded into miss by CalculateMisses); single-hidden XOR convergence regression preserved.
- [x] **T-5A07** — `pkg/network/network_test.go` + `propagation_test.go`: add table-driven tests for two-hidden and three-hidden chains with hand-computed reference vectors (analogous to `pkg/compute/cpu/cpu_test.go` golden math); confirm single-hidden regression suite still passes; `go test -race -cover` with ≥ 80 % coverage on `pkg/network`. **Changes:** New `propagation_test.go` with chain wiring table test, golden forward + golden backward (2-hidden, 3-hidden Linear), 2-hidden Sigmoid XOR convergence; `pkg/network` coverage 95.8 %, race-clean.

## Track B — Public Facade Lift (`pkg/nn`)

**Spec:** [l2-multihidden-impl.md](../specifications/l2-multihidden-impl.md) §5.5
**Depends on:** Track A complete (Network[T] generalisation must land first; `compile()` lift is meaningless against the old single-bundle storage).

- [ ] **T-5B01** — `pkg/nn/compile.go`: remove the `if len(cfg.HiddenLayers) > 1 { return ErrUserConfig "multi-hidden networks not supported in v0.1 ..." }` gate. Build `hiddens := []*layer.Dense[T]` from `cfg.HiddenLayers` and pass via the new variadic `SetLayers`. Existing per-hidden Size / Activation validation (already in `validate()`) is unchanged.
- [ ] **T-5B02** — `pkg/nn/compile.go` `emitSoftWarnings`: confirm the existing "deep stack with WeightInitRandom" Warn fires on `len(cfg.HiddenLayers) > 5 && cfg.WeightInit == WeightInitRandom`. The branch already exists; this task adds a Logger.Warn assertion test that exercises it with seven hidden layers.
- [ ] **T-5B03** — `pkg/nn/nn_test.go`: multi-hidden Compile + Fit tests for two-hidden, three-hidden, and seven-hidden chains. Single-hidden XOR convergence test stays in place as the "no-regression" baseline. `PresetMNIST` and `PresetRegression` Compile() now succeeds (v0.1 returned ErrUserConfig); add positive-path test for both. `go test -race -cover` ≥ 80 % on `pkg/nn`, no existing test regresses.

## Track C — Persistence Schema Bump (`pkg/persistence`)

**Spec:** [l2-multihidden-impl.md](../specifications/l2-multihidden-impl.md) §5.4
**Depends on:** Track B (a multi-hidden network must compile before its weights can be round-tripped). Independent of Track D.

- [ ] **T-5C01** — `pkg/persistence/config.go`: bump `SchemaVersion` constant `"1.0.0"` → `"1.1.0"`. Update spec comment block to reflect v0.2 schema. Forward-compatibility per [l1-network-persistence.md](../specifications/l1-network-persistence.md) §3 PERS-1: minor mismatch is warning + best-effort load, so v0.1 readers loading a v0.2 file degrade to a non-fatal warning.
- [ ] **T-5C02** — `pkg/persistence/persistence_test.go`: regression test loading a v1.0.0 fixture (previous output of `examples/persistence/`) under the new code — must succeed (PERS-1 minor mismatch warning). Add positive multi-hidden round-trip: train a two-hidden network in test, extract via `extractWeights`, write, read back, assert post-reload Query within `cpu.ToleranceF32`. ≥ 80 % coverage maintained.
- [ ] **T-5C03** — `examples/persistence/main.go`: extend the `extractWeights` and `installWeights` helpers (currently hard-coded for two layers — hidden_0 + output) to walk `n.Network.Hiddens` slice. Update `buildConfigDoc` to emit the multi-hidden HiddenLayers list. Smoke test sets up a two-hidden topology and asserts the round-trip is bit-identical for `float64` and ≤ ToleranceF32 for `float32`.

## Track D — Catalog Examples (`examples/`)

**Spec:** [l2-usage-examples.md](../specifications/l2-usage-examples.md) §5.2 / E03..E13 (excluding E06 / E10 — see Deferred section below)
**Depends on:** Track B (compile() must accept multi-hidden). Independent of Track C unless the example exercises persistence (E13 does not).

- [ ] **T-5D01** — `examples/perceptron/` (E03): restore the spec-canonical 4-hidden topology `3 → Sigmoid(5) → ReLU(10) → Sigmoid(5) → SoftMax(2)`, mixed bias per spec. rate=0.3, loss=ARCTAN, max-iter=100000, loss-limit=1e-6. Builder API. Drop the `// removed once v0.2 lands` doc-comment. Smoke test asserts the published reference query `[-0.52, 0.66, 0.81] → ≈ [-0.13, 0.2]` within a loose tolerance (random init drift).
- [ ] **T-5D02** — `examples/binary_classification/` (E04): new module with 2 → ReLU(8) → ReLU(8) → Sigmoid(1), bias on; 200-point Gaussian-blob dataset generated in code. rate=0.01, loss=BinaryCrossEntropy (loss.BCE), init=he, max-iter=2000. Options API. Hold-out 20 % split; smoke test asserts train accuracy > 90 %, test accuracy > 85 %.
- [ ] **T-5D03** — `examples/iris/` (E05): new module with 4 → ReLU(16) → ReLU(8) → SoftMax(3), bias on; ship Iris CSV under `examples/iris/data/iris.csv` (~5 KiB, public-domain). rate=0.01, loss=CrossEntropy, init=he, max-iter=3000. Options API with `WithEpochCallback` printing every 100 epochs. Smoke test asserts test-set accuracy ≥ 85 % (loose vs spec 90 % to absorb init drift).
- [ ] **T-5D04** — `examples/regression_sin/` (E07): new module with 1 → TanH(16) → TanH(16) → Linear(1), bias on; 200-point grid x ∈ [−π, π], y = sin(x). rate=0.01, loss=MSE, init=xavier, max-iter=5000. Builder API. Smoke test asserts RMSE ≤ 0.10 on a fresh held-out grid.
- [ ] **T-5D05** — `examples/regression_multi/` (E08, v0.2 adapted): new module with 5 → ReLU(16) → ReLU(8) → Linear(3); generated 5-dim → 3-dim known linear+noise function. rate=0.01, loss=MSE, max-iter=3000. Options API (skip `PresetRegression` extension — that's a v0.3 follow-up; spec note added). Per-output-dim RMSE printed; smoke test asserts every dim RMSE ≤ 0.20.
- [ ] **T-5D06** — `examples/higher_order_options/` (E13): new module exercising `Sequential` and `DeepNetwork` higher-order options. Topology A `4 → Sequential(3, 16, ReLU) → SoftMax(3)` (three identical hidden layers). Topology B `4 → DeepNetwork(32, 4, ReLU) → SoftMax(3)` (sizes 32 → 16 → 8 → 4). Reuses Iris CSV from E05. Smoke test asserts both reach ≥ 80 % test accuracy.
- [ ] **T-5D07** — `examples/README.md`: move E03..E13 (six entries) from `## Deferred to v0.2` to `## v0.1 Active` (renamed `## v0.1 / v0.2 Active`); refresh the Coverage Matrix Audit — close gaps for `Sequential`, `DeepNetwork`, `PresetMNIST` (still gap until E06 lands), `PresetRegression` (still gap; deferred), `Verify` (now closed by E04 / E05 / E07). Add a v0.2 release notes line summarising the new active set. Update `go.work` with the six new modules.

## Phase Gate (T-5Z)

- [ ] **T-5Z01** — `go build ./...` plus per-module `go build ./examples/{xor,style_showcase,logic_gates,callbacks,persistence,shared_options,precision,perceptron,binary_classification,iris,regression_sin,regression_multi,higher_order_options}/...` — green across library + 13 example modules.
- [ ] **T-5Z02** — `go test -cover ./...` plus per-module example tests — every touched library package retains ≥ 80 % line coverage; new examples ship with smoke tests; existing Phase 1-4 baselines do not regress.
- [ ] **T-5Z03** — `go test -race ./...` plus per-module example races — race-clean (run via PowerShell for gcc PATH per existing 2026-04-29 STATE.md note). Phase 4 `TestPauseResumeCycle` remains pre-existing flake — non-blocking, unchanged scope.
- [ ] **T-5Z04** — STATE.md updated; CHANGELOG.md Phase 5 entry; PLAN.md Phase 5 → ✓ Done; `examples/README.md` v0.2 release notes; v0.2 release-ready bar. Outstanding v0.2 backlog (E06 MNIST loader spec, E10 AndTrain API surface, `PresetRegression` extension) carried forward to a future phase.

## Track Execution Order

```mermaid
graph LR
    A[Track A — Network storage + propagation] --> B[Track B — pkg/nn compile() lift]
    B --> C[Track C — persistence schema 1.1.0]
    B --> D[Track D — six new / refactored examples]
    C --> Z[Gate T-5Z]
    D --> Z
```

Strict serial A → B (B depends on A's slice generalisation). C and D run in parallel after B. Gate validates the whole stack.

## Risk Audit (`@role:planning-skeptic`)

- **Cascade risk**: Track A is the single point of failure. Every other track is gated on Network[T]'s slice generalisation. Mitigation: T-5A07 lands the regression suite alongside the storage change, so any propagation bug surfaces before B/C/D can start.
- **Optimism bias**: Six new examples (T-5D02..D06) each ship dataset + smoke test + module bootstrap. Estimated ~30 min each on a familiar facade — realistic given the Phase 4 v0.1 cadence (seven examples in one session). E03 is mostly a topology restoration; not new work.
- **Hidden dependency — persistence wire format**: bumping `SchemaVersion` to "1.1.0" is forward-compatible per PERS-1 but the v0.1 fixture in `examples/persistence/` re-runs every test session and will emit a "minor schema mismatch" warning post-bump. T-5C02 includes the regression load to confirm warning ≠ hard error.
- **Cascade risk — `PresetMNIST` / `PresetRegression`**: spec says Compile() succeeds after the lift. Both presets internally configure multi-hidden. Phase 5 does NOT ship MNIST / Regression-data examples (E06 / E08 dataset-loader still gated for E06; E08 substituted with synthetic data in T-5D05) — T-5B03 only asserts Compile() / Fit() smoke, not full training accuracy on real data.

## Deferred to Phase 6+

- **E06 MNIST preset** — needs MNIST dataset-loader spec authored separately (post-Phase-5 magic.spec invocation).
- **E10 Continuation** — needs `AndTrain` API surface specified (l2-nn-facade extension or new spec).
- **`PresetRegression` parameter extension** — per spec §5.2 / E08 the preset is incomplete for the multi-hidden case; resolve in a follow-up minor.
