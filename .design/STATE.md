# Project State

<!-- STATE.md — live project memory. Read FIRST in every workflow session. -->
<!-- Maximum 100 lines. Agent updates AFTER each completed action. -->

**Workspace:** main
**Project Version:** 0.7.0 (Phase 8 complete)
**Updated:** 2026-05-10 22:00
**Phase:** 8 — LR Scheduling Extension + CLI Binary
**Status:** Done

## Current Position

- **Task:** Phase 8 complete. All 9 tasks Done (T-8A01, T-8B01..T-8B05, T-8T01, T-8T02, T-8Z01).
- **Spec:** l2-lr-scheduling-impl.md bumped to v1.1.0 (ExponentialLR in Implemented). Gate T-8Z01 passed.
- **Next Action:** Create v0.7.0 git tag per l1-release-policy.md §5.4; plan v0.8.0 scope.

## Progress

```
Phase 1 (Done):   [22/22]   ████████ 100%
Phase 2 (Done):   [26/26]   ████████ 100%
Phase 3 (Done):   [19/19]   ████████ 100%
Phase 4 (Done):   [14/14]   ████████ 100%
Phase 5 (Done):   [23/23]   ████████ 100%   (all tracks + gate complete)
Phase 6 (Done):   [19/19]   ████████ 100%   (all tracks + validation + gate + tag)
Phase 7 (Done):   [16/16]   ████████ 100%   (Tracks A+B+C + validation + gate)
Phase 8 (Done):   [9/9]     ████████ 100%   (Track A: ExponentialLR + Track B: CLI binary + gate)
Overall:          [148/148] ████████ 100%
```

## Recent Decisions

- 2026-05-10 **Decision:** Phase 8 complete. Track A: `pkg/optimizer/exponential_lr.go` — `ExponentialLR[T]` (lr₀ × gamma^t), LRS-1..LRS-6, default PerEpoch, JSON SaveState/LoadState; `pkg/optimizer/` coverage 88.2%. Track B: `cmd/gonn/` — `main.go` (dispatch), `train.go`, `query.go`, `verify.go`, `version.go`, `exitcode.go`, `load.go`, `csv.go` — full CLI binary with train/query/verify/version subcommands, `--precision float32|float64` generic dispatch, 64 MB streaming threshold, `--json` output, exit-code contract (0-7), XOR end-to-end smoke test; coverage 83.7%. l2-lr-scheduling-impl.md bumped to v1.1.0. CHANGELOG.md v0.7.0 entry written. Gate T-8Z01: `go build ./...` clean; `go test ./...` all 16 packages green; all packages ≥80%.
- 2026-05-10 **Decision:** Phase 7 complete. Track A: `pkg/optimizer/` extended with `Scheduler[T]` interface, `BindScheduler[T]`, `LearningRateSetter[T]` optional extension, and four scheduler types — `StepLR[T]` (step decay), `WarmUpLR[T]` (linear ramp, PerStep default), `CosineAnnealingLR[T]` (cosine decay), `ChainScheduler[T]` (sequential composition). All four existing optimizers (SGD/Adam/RMSProp/SGDMomentum) implement `LearningRateSetter[T]` via `SetLearningRate(T)`. `pkg/optimizer/scheduler_test.go` covers all scheduler types, BindScheduler wiring, Granularity defaults, and SaveState/LoadState round-trips; coverage 88.9 %. Track B: `pkg/nn/builder.go` + `pkg/nn/options.go` extended with bulk constructors `Repeat`/`Pattern`/`HiddenLayers` (Builder, setter/append semantics distinguished) and `Repeat[T]`/`Pattern[T]`/`WithHiddenLayers[T]` (Options, append); `WithScheduler` added to both; `pkg/nn/train.go` dispatches `sched.Step()` per `Granularity()` (PerEpoch after epoch, PerStep per batch); `pkg/nn/config.go` adds `Scheduler` field; `pkg/nn/phase7_test.go` covers 11 test functions + 2 benchmarks; coverage 86.4 %. Track C: `skills/gonn/` created with `SKILL.md` (10 sections, YAML frontmatter), 3 example files (builder-xor, options-mnist, deep-network), and 2 resource files (api-reference, conventions). Gate T-7Z01: `go build ./...` clean; `go test ./pkg/...` all green; all packages ≥80 %; skills directory matches spec §5.1; 3 orphaned specs resolved.
- 2026-05-08 **Decision:** Phase 6 complete. Tracks A+B (optimizer + regularizer + WeightInit fix): `pkg/optimizer/` (SGD/Adam/RMSProp/SGDMomentum, 98.9% cover, 0 allocs/op benchmarks), `pkg/regularizer/` (L1/L2/Dropout/Compose, 80.0% cover), `pkg/nn` wired via `WithOptimizer`/`WithRegularizer`/`trainStep()`. Track C: CHANGELOG.md v0.6.0, README.md updated with full v0.6 API docs. Validation: TestWeightInitRanges, TestRegularizerConvergence, TestInferenceNoDropout, TestOptimizerIntegration all green. Gate T-6Z01: all packages ≥80% cover (`network` 96.3%, `nn` 86.4%). Tag v0.6.0 created. Pre-existing TestPauseResumeCycle timing flake documented in CHANGELOG Known Issues.

## Blockers

- (none — Phase 7 complete)

## Blocking Constraints

- (none — all tracks green, phase gate passed)
- Note: race detector via PowerShell only on Windows (gcc PATH issue, pre-existing).

## Session Continuity

**Last Session Ended:** 2026-05-10
**Handoff File:** none
**Bootstrap Mode:** false (Phase 7 complete; next = plan v0.7.0 scope)
