# Project State

<!-- STATE.md — live project memory. Read FIRST in every workflow session. -->
<!-- Maximum 100 lines. Agent updates AFTER each completed action. -->

**Workspace:** main
**Project Version:** 0.1.0
**Updated:** 2026-05-02
**Phase:** 4 — Examples Catalog (Active — decomposed, ready for /magic.run)
**Status:** Active

## Current Position

- **Task:** T-4A01 (Pending) — Phase 4 Track A first task
- **Spec:** l2-usage-examples promoted RFC → Stable v1.0.0; INDEX.md 2.0.0 → 2.1.0; PLAN.md 1.4.0 → 1.5.0; TASKS.md 1.4.0 → 1.5.0; phase-4.md decomposed.
- **Next Action:** Run `/magic.run` to execute Phase 4. Track execution order: A first (smoke-test pattern), then B+C+D in parallel; E after all examples land; Gate T-4Z01..T-4Z04 closes phase.

## Progress

```
Phase 1 (Done):   [22/22]  ████████ 100%
Phase 2 (Done):   [26/26]  ████████ 100%
Phase 3 (Done):   [19/19]  ████████ 100%
Phase 4 (Active): [0/14]   ░░░░░░░░  ~0% (10 feature + 4 gate)
Overall:          [67/81]  ███████░  ~83% of v0.1 MVP
```

## Recent Decisions

- 2026-05-02 **Decision:** Phase 4 activated and decomposed via /magic.task update. l2-usage-examples promoted RFC → Stable v1.0.0 (E09 ungated; persistence Stable since 2026-05-01). 14 atomic tasks across Tracks A–E + 4 gate checks. v0.1 scope is 7 single-hidden examples (E01, E02, E09, E11, E12, E14-adapted, E15); 8 multi-hidden / AndTrain / MNIST entries deferred to v0.2 backlog. INDEX.md 2.0.0 → 2.1.0; PLAN.md 1.4.0 → 1.5.0; TASKS.md 1.4.0 → 1.5.0. RULES.md v1.2.0 parity confirmed.
- 2026-05-01 **Decision:** Phase 3 closed via /magic.run. All 5 tracks green: persistence (81.4 % cover), checkpoint (83.2 %), dataset (87.6 %), compute (97.3 %), compute/cpu (100 %), network (96.1 %), nn (83.7 %). PERF-4 backward-pass benches at 0 allocs/op. pprof opt-in via `WithProfiling[T](addr)`. Phase Gate T-3Z01..T-3Z04 all green; CHANGELOG entry added; PLAN 1.3.0 → 1.4.0; TASKS 1.3.0 → 1.4.0. Phase 4 unblocked.
- 2026-05-01 **Decision:** Phase 3 decomposed via /magic.task update. PLAN.md 1.2.0→1.3.0; TASKS.md 1.2.0→1.3.0. 15 atomic tasks across Tracks A–E + 4 gate checks. Track ordering: A→B serial; C,D parallel with A; E last. RULES.md v1.2.0 parity confirmed.
- 2026-05-01 **Decision:** Batch Stabilization complete. Promoted 9 L1 RFC → Stable 1.0.0 and 5 L2 Draft → Stable 1.0.0. Phase 3 unblocked. INDEX.md 1.9.0 → 2.0.0. C9 Trust Mode auto-promotion via MVC criteria.
- 2026-04-30 **Decision:** Phase 2 complete. `pkg/nn` facade landed: NewBuilder + New[T](opts...), Compile/MustCompile, Train/Fit/Query/Verify, Pause/Resume/Stop. 84.8% coverage, race-clean. l2-nn-facade promoted RFC → Stable v2.0.0; l2-training-loop + l2-control-impl Draft → Stable v1.0.0. Multi-hidden topology (PresetMNIST/Regression) errors at Compile() with v0.2 deferred-feature note — this constraint shapes Phase 4 v0.1 scope.
- 2026-04-29 **Decision:** `go test -race ./...` green on every Phase-1 package. Run via PowerShell because the Claude Code bash-shell does not propagate Windows PATH to the Go child process (gcc lives at C:/msys64/mingw64/bin). Documented for repeatability.

## Blockers

- (none — Phase 4 ready for /magic.run; v0.2 backlog cleanly partitioned)

## Blocking Constraints

- **Multi-hidden v0.2** — `compile()` rejects `len(HiddenLayers) > 1`. Active during Phase 4: 8 catalog entries (E03, E04, E05, E06, E07, E08, E10, E13) deferred to v0.2 backlog rather than blocking Phase 4 v0.1.
- (race detector confirmed green via PowerShell on 2026-05-01; pre-existing TestPauseResumeCycle flake on `pkg/nn` reproduces on `develop` and is unrelated to Phase 3 work; non-blocking)

## Session Continuity

**Last Session Ended:** 2026-05-02
**Handoff File:** none
**Bootstrap Mode:** false (all Phase-1, Phase-2, Phase-3 specs Stable; l2-usage-examples Stable; Phase-4 ready for /magic.run)
