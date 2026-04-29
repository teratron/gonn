# Changelog

All notable changes to the GoNN library will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
release artifacts dictated by [.magic/run.md](.magic/run.md) Phase Completion / Plan Completion.

## [Unreleased]

### Phase 1 Track A — 2026-04-29 [Bootstrap]

Foundations slice of the Phase 1 Foundation Rewrite. Closes the `pkg/utils`
half of blocker C-001 (Tracks B/D still pending). Specs `l2-errors-impl` and
`l2-init-impl` are RFC; promotion to Stable deferred to Phase Gate T-1Z02.

#### Added

- `pkg/utils/errors.go` — six orthogonal category sentinels (`ErrUserConfig`,
  `ErrInputData`, `ErrCompute`, `ErrControl`, `ErrIntegrity`, `ErrIO`) plus
  `Newf`, `Wrap`, `NewSizeError`, `NewActivationError`, `NewIntegrityError`,
  `LocationHint`. Multi-`%w` wrapping preserves both category and cause for
  `errors.Is` routing.
- `pkg/utils/init.go` — `NewRNG(seed) (*rand.Rand, uint64)` based on
  `math/rand/v2.PCG`; generic samplers `XavierUniform[T]`, `HeNormal[T]`,
  `Uniform[T]` over `utils.Float`. Zero-allocation hot path (~15-42 ns/op).
- `pkg/utils/errors_test.go`, `pkg/utils/init_test.go` — 100 % line coverage,
  C32 forbidden-phrase sweep, distribution + reproducibility checks,
  zero-alloc benchmarks via `b.Loop()`.

#### Changed

- `.design/specifications/l2-errors-impl.md` — bumped 0.2.0 → 0.3.0;
  sentinel set finalized to 6 orthogonal categories. `ErrTrainingFailure`
  and `ErrUnsupported` from v0.2.0 dissolved into `ErrCompute` /
  `ErrUserConfig` to avoid catch-all routing.
- `.design/INDEX.md`, `.design/PLAN.md` — registry version aligned.
- `.design/tasks/phase-1.md` — `[Bootstrap]` markers added to T-1A0x;
  Track A frontmatter populated with provides / key_files / patterns_established.

#### Notes

- `go test -race` skipped in dev environment (no gcc). Race-detector pass
  is recorded as a Phase Gate requirement (T-1Z02) and must run on a
  CGO-enabled host before merging.

### Phase 1 Track B — 2026-04-29

Neuron slice of the Phase 1 Foundation Rewrite. Closes the `pkg/neuron`
half of blocker C-001: `Hidden[T]` exists, `Output.CalculateValue` no
longer recurses, `axon.OutgoingCell` is restored. `pkg/network/...` still
fails to build — that surface is owned by Tracks C/D.

#### Added

- `pkg/neuron/cell/hidden.go` — `type Hidden[T utils.Float] = Dense[T]`
  generic alias plus `NewHidden(number)` constructor. Closes the missing
  symbol referenced by `pkg/network/network.go`.
- `pkg/neuron/axon` — `NewWithWeight(weight, incoming, outgoing)` for the
  layer-driven path that pre-samples weights via Xavier/He (Track A).
- `pkg/neuron/cell/cell_test.go`, `pkg/neuron/axon/axon_test.go` — table-
  driven contract tests, 100 % line coverage, regression test that pins
  the Output non-recursion property, concurrency probe over the package
  PCG mutex.

#### Changed

- `pkg/neuron/cell/core.go`, `input.go`, `bias.go`, `dense.go`,
  `output.go` — full C31 doc-comments; `_NewBias` renamed `NewBias`;
  `Output.CalculateValue` rewritten to delegate to embedded Dense and
  compute residual `target - value` only when target is non-nil.
- `pkg/neuron/axon/axon.go` — restored `OutgoingCell neuron.Neuron[T]`;
  swapped `math/rand` legacy global for `math/rand/v2.PCG` via
  `utils.NewRNG(0)`; `New` and `NewWithWeight` distinguish default-init
  from caller-controlled paths.

### Phase 1 Track C — 2026-04-29

Layer slice of the Phase 1 Foundation Rewrite. Closes the nil-deref
constructors, the duplicate Init logic, and the dropped-size defect from
[l2-layer-types] §5.3.

#### Added

- `pkg/layer/layer_test.go` — 93.7 % line coverage. Includes regression
  tests for nil-deref panic, dropped size on `NewInput`, Init de-dup,
  bias-cell alloc, target pointer aliasing.

#### Changed

- `pkg/layer/core.go`, `base.go`, `input.go`, `dense.go`, `output.go` —
  full rewrite. Constructors now allocate the embedded core/base via
  `newCore` / `newBase`; `Dense.Init` delegates to `base.Init`;
  `NewInput(size)` honours the size argument; bias cells allocated when
  `Bias=true`; Output owns a `targets []T` slice and exposes
  `SetTarget(idx, value)` for label updates between samples.

### Phase 1 Track D — 2026-04-29

Network slice of the Phase 1 Foundation Rewrite. Closes blocker C-001
end-to-end: `go build ./...` is green and the XOR smoke test converges
within 5000 epochs (sigmoid 2-4-1, MSE).

#### Added

- `pkg/network/network_test.go` — 96.5 % line coverage. XOR-convergence
  smoke test (target loss < 0.02), Build axon-fan-out, SetInputs writes
  every cell (regression of [l2-network-graph] §5.4 #4), error routing
  through ErrUserConfig / ErrInputData.
- `Network.Train(input, target)` — one-shot forward + backward + update
  helper used by the smoke test and by Phase-2 facade.
- `Network.LossMode()`, `Network.CalculateLossDefault()` — surfaces the
  loss configured by SetLayers.

#### Changed

- `pkg/network/network.go` — `New[T]()` returns Network by value (kept
  for [pkg/nn] embed compatibility); `SetLayers(in, hidden, out)`
  installs cells from layer constructors and captures activation / loss
  / bias-cell handles; `Build()` wires Input→Hidden and Hidden→Output
  axons (and bias-axon when configured); `SetInputs` / `SetTargets`
  write per-cell instead of cells[0].
- `pkg/network/bundle.go` — slimmed to `cells []S` + `Replace`, `Add`,
  `Cells`, `At`, `Len`. The legacy whole-slice `any(...).([]Neuron[T])`
  cast (always false) is replaced by element-wise type assertion in
  callers (closes [l2-network-graph] §5.4 #3).
- `pkg/network/propagation.go` — forward applies layer activation via
  the [pkg/activation] dispatcher; pre-activation values cached in
  `Network.preactHidden / preactOutput` so backprop feeds pre-σ to
  `activation.Derivative` (needed for sigmoid σ' = σ(x)·(1-σ(x)));
  backward walks Output.Axons in reverse to credit Hidden cells.

### Phase 1 Gate — 2026-04-29

- T-1Z01 `go build ./...` — green. C-001 closed.
- T-1Z02 `go test -cover ./...`:
  - pkg/utils 100 %
  - pkg/neuron/cell 100 %
  - pkg/neuron/axon 100 %
  - pkg/layer 93.7 %
  - pkg/network 96.5 %
  - pkg/activation 9.1 % (Phase 0 baseline; out of scope)
  - pkg/loss 13.5 % (Phase 0 baseline; out of scope)
  - pkg/nn 0 % (Phase 2 facade; out of scope)
- T-1Z02 `go test -race ./...` — **PASS** on all 7 packages
  (utils, neuron/cell, neuron/axon, layer, network plus pre-existing
  activation, loss). Confirmed under MSYS2 mingw64 gcc 15.2.0 via
  PowerShell. Note: the Claude Code bash shell does not propagate
  Windows PATH to the Go child process; race-detector runs must be
  launched from PowerShell or `cmd.exe` until that environment is
  fixed.
- T-1Z03 STATE.md cleared blocker C-001; phase-1 frontmatter populated
  with provides / key_files / patterns_established / status:Done.
