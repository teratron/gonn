# Project State

<!-- STATE.md — live project memory. Read FIRST in every workflow session. -->
<!-- Maximum 100 lines. Agent updates AFTER each completed action. -->

**Workspace:** main
**Project Version:** 0.1.1
**Updated:** 2026-05-04 13:52
**Phase:** 5 — Multi-Hidden Topology (v0.6) (Active — Tracks A+B green; Tracks C+D next)
**Status:** Active

## Current Position

- **Task:** T-5B03 Track B multi-hidden Compile+Fit tests
- **Spec:** l2-multihidden-impl Stable v1.0.0; INDEX.md 2.3.0; PLAN.md 1.7.0; TASKS.md 1.7.0; phase-5.md Tracks A + B all `[x]`.
- **Next Action:** Tracks C + D in parallel (persistence schema bump + 6 catalog examples)

## Progress

```
Phase 1 (Done):   [22/22]  ████████ 100%
Phase 2 (Done):   [26/26]  ████████ 100%
Phase 3 (Done):   [19/19]  ████████ 100%
Phase 4 (Done):   [14/14]  ████████ 100%
Phase 5 (Active): [10/23]  ███░░░░░  ~43%   (Tracks A+B complete; C+D pending)
Overall:          [91/104] █████████ ~88%   v0.5 closed; v0.6 in flight
```

## Recent Decisions

- 2026-05-04 **Decision:** Track B landed. `pkg/nn.compile()` gate `if len(cfg.HiddenLayers) > 1 { return ErrUserConfig ... }` removed; `compile()` now builds the full `[]*layer.Dense[T]` chain from `cfg.HiddenLayers`. Outdated tests flipped to v0.6 positive paths: `TestBuilderAcceptsMultiHidden`, `TestPresetMNISTCompiles`, new `TestPresetRegressionCompiles`. New `multihidden_test.go` covers (a) `TestDeepStackRandomInitWarn` + Xavier negative control via `captureWarnings` slog-buffer helper, (b) `TestCompileFitMultiHiddenChainDepths` for depths {2,3,7}, (c) `TestCompileMultiHiddenTwoHiddenConverges` (Sigmoid Xavier, 20000 epochs, ≤ 0.10). pkg/nn coverage 85.2 %, race-clean. **Track A correction**: initial `CalculateMisses` stored δ on miss (per spec §5.6 pseudocode), but that altered the v0.5 single-hidden ΔW arithmetic enough that `examples/callbacks` `TestTrainConverges` (XOR loss < 0.15 in 2000 epochs) regressed. Reverted to v0.5 pattern — raw miss in `CalculateMisses`, derivative folded inside `CalculateWeights` via `eff = rate × σ'(z)` per layer — extending positionally to every chain entry. Single-hidden behaviour bit-identical to v0.5; multi-hidden chain extension is the v0.5 omission propagated layer-by-layer (faster initial gradients than fully-correct backprop, but matches `T-5A06`'s explicit "preserves v0.5 single-layer arithmetic when len == 1"). All `pkg/...` and 7 v0.5 example modules green under `-race`.
- 2026-05-03 **Decision:** Track A landed. `pkg/network.Network[T]` storage now slice-shaped (`Hiddens []bundle`, `hiddenBiases`, `hiddenActs`, `preactHiddens` parallel to it); `SetLayers` accepts `[]*layer.Dense[T]`; Build/CalculateValues/CalculateMisses/CalculateWeights walk the chain per [l2-multihidden-impl] §5.3 / §5.6. New `propagation_test.go` covers chain wiring (table test), forward + backward goldens (Linear 2-hidden, 3-hidden), and Sigmoid 2-hidden XOR convergence. Coverage 95.8 %, race-clean. Downstream callers (`pkg/nn.compile`, `pkg/nn/train.go` snapshot/restore/weightCount, `pkg/nn/nn_test.go`, `examples/persistence`) migrated to slice-form access.
- 2026-05-03 **Decision:** Phase 5 activated and decomposed via /magic.task update. l2-multihidden-impl promoted Draft → Stable v1.0.0 (Trust Mode — MVC + Implements Stable + only scoped TBDs). 19 atomic tasks across Tracks A–D + 4 gate checks. Track A → B serial (storage generalisation must precede compile() lift); C and D parallel after B. Six v0.6 catalog entries (E03/E04/E05/E07/E08/E13) promoted from Phase 4 backlog into Phase 5. E06 (MNIST loader) and E10 (AndTrain) stay deferred. INDEX.md 2.2.0 → 2.3.0; PLAN.md 1.6.0 → 1.7.0; TASKS.md 1.6.0 → 1.7.0.
- 2026-05-03 **Decision:** New v0.6 anchor spec `l2-multihidden-impl` Draft v0.5.0 authored via /magic.spec Proactive Architect mode. Captures three coordinated deltas — `pkg/network.Network[T]` storage chain, `pkg/nn.compile()` gate lift, `pkg/persistence` weights schema 1.0.0 → 1.1.0. INDEX.md 2.1.0 → 2.2.0.
- 2026-05-02 **Decision:** Phase 4 closed via /magic.run. 7 example modules build, test, race-clean. Coverage matrix audit at examples/README.md flags 6 v0.6-gated API surfaces. Phase Gate T-4Z01..T-4Z04 green. v0.5 release-ready bar reached. Established conventions: per-example go.mod with replace directive; `runX()` helpers extracted from `main()` for smoke tests; pkg/nn ↔ pkg/persistence seam documented in E09.
- 2026-05-02 **Decision:** Phase 4 activated and decomposed via /magic.task update. l2-usage-examples promoted RFC → Stable v1.0.0 (E09 ungated; persistence Stable since 2026-05-01). 14 atomic tasks across Tracks A–E + 4 gate checks scoped to v0.5's single-hidden constraint.
- 2026-05-01 **Decision:** Phase 3 closed via /magic.run. All 5 tracks green: persistence (81.4 %), checkpoint (83.2 %), dataset (87.6 %), compute (97.3 %), compute/cpu (100 %), network (96.1 %), nn (83.7 %). PERF-4 backward-pass at 0 allocs/op. pprof opt-in via `WithProfiling[T](addr)`.
- 2026-04-30 **Decision:** Phase 2 complete. Multi-hidden topology errors at Compile() with v0.6 deferred-feature note — this constraint shapes Phase 4 v0.5 scope and is now lifted by Phase 5.
- 2026-04-29 **Decision:** `go test -race ./...` green via PowerShell because the Claude Code bash-shell does not propagate Windows PATH to the Go child process (gcc lives at C:/msys64/mingw64/bin). Documented for repeatability.

## Blockers

- (none — Tracks A + B complete; C + D unblocked, can run in parallel)

## Blocking Constraints

- **Track A cascade gate cleared**: T-5A01..A07 all landed; multi-hidden chain proven by table-driven golden math + Sigmoid XOR convergence. Backward arithmetic preserves v0.5 single-hidden ΔW exactly.
- **Track B gate lift complete**: `pkg/nn.compile()` accepts any positive `len(HiddenLayers)`; multi-hidden Compile + Fit smoke-tested at depths 2, 3, 7. Tracks C + D unblocked.
- **Persistence forward-compat**: SchemaVersion 1.0.0 → 1.1.0 is minor — v0.5 readers loading v0.6 files emit a warning per PERS-1, not a hard error. T-5C02 covers the regression load.
- (race detector via PowerShell only — pre-existing TestPauseResumeCycle flake on `pkg/nn` reproduces on `develop` and is unrelated; non-blocking)

## Session Continuity

**Last Session Ended:** 2026-05-03
**Handoff File:** none
**Bootstrap Mode:** false (l2-multihidden-impl Stable; Phase 5 ready for /magic.run)
