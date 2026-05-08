---
phase: 7
name: Deep Builder + LR Scheduling + Developer Skills
status: Todo
subsystem: pkg/nn, pkg/optimizer, skills/gonn
requires:
  - phase-6 (optimizer + regularizer + v0.6.0 release complete)
provides: []
key_files: []
patterns_established: []
duration_minutes:
---

# Phase 7 — Deep Builder + LR Scheduling + Developer Skills

**Status:** Todo
**Decomposed:** 2026-05-08
**Tasks:** 13 feature + 2 validation + 1 gate = 16 total
**Specs:** l1-lr-scheduling v1.0.0, l2-deep-builder v1.0.0, l2-gonn-skills v1.0.0
**Track order:** A (L1 first) → B (L2, after A); C (L2, parallel with A+B); D (L2, parallel with A+B+C);
                 T-7T01 after A+B; T-7T02 after C+D; Gate T-7Z after all tracks.

## Track A — Learning Rate Scheduling (L1 Contract → pkg/optimizer extension)

*Goal: `pkg/optimizer/` extension with Scheduler interface + StepLR / CosineAnnealingLR / WarmUpLR / ChainScheduler.*
*Source: [l1-lr-scheduling.md](../specifications/l1-lr-scheduling.md)*

- [ ] **T-7A01** — Define `pkg/optimizer/scheduler.go`: `Scheduler[T Float]` interface with
  `Step() T`, `Reset()`, `Granularity() Granularity`, `SaveState()/LoadState()`. Define
  `Granularity` enum (`PerEpoch`, `PerStep`). Define `BindScheduler(opt, sched)` function
  that wires the scheduler to the optimizer's internal rate.

- [ ] **T-7A02** — Implement `pkg/optimizer/step_lr.go`: `StepLR[T]` — rate decays by `gamma`
  every `stepSize` steps. Formula: lr₀ × gamma^(⌊t / stepSize⌋). LRS-1..LRS-4 compliance.
  Tests: step boundary, gamma=1 no-op, gamma=0 zero-rate, SaveState/LoadState round-trip.

- [ ] **T-7A03** — Implement `pkg/optimizer/warmup_lr.go`: `WarmUpLR[T]` — linear ramp from 0
  to lr₀ over `warmupSteps` steps. After warmup, holds constant. LRS-2 (no mod until Step)
  and LRS-4 (hold after expiry) compliance.

- [ ] **T-7A04** — Implement `pkg/optimizer/cosine_lr.go`: `CosineAnnealingLR[T]` — cosine
  decay from lr₀ to `lrMin` over `T_max` steps. Formula: lrMin + 0.5(lr₀ − lrMin)(1 + cos(πt/T_max)).
  LRS-4 compliance: hold `lrMin` after `T_max`.

- [ ] **T-7A05** — Implement `pkg/optimizer/chain_scheduler.go`: `ChainScheduler[T]` — takes
  `[]SchedulerSegment{Scheduler, Duration}`, runs each sub-scheduler for its duration window.
  Global step counter. LRS-3 composition contract. Reset propagation to sub-schedulers.

- [ ] **T-7A06** — Wire scheduler into `pkg/nn/train.go`: after each epoch (or step, based on
  `Granularity()`), call `scheduler.Step()`. Add `WithScheduler(sched Scheduler[T])` option
  to `pkg/nn/options.go`. If no scheduler provided, training loop is unchanged (nil-guard).

## Track B — Deep Network Builder Ergonomics (pkg/nn extension)

*Goal: Repeat/Pattern/HiddenLayers bulk constructors in both Builder and Options API styles.*
*Source: [l2-deep-builder.md](../specifications/l2-deep-builder.md)*
*Requires: Track A concepts proven (LR scheduler pairs with deep networks).*

- [ ] **T-7B01** — Add `Repeat(count, size uint, act activation.Type, bias bool) *NN[T]`
  builder method to `pkg/nn/builder.go`. Appends `count` identical `HiddenLayerSpec[T]`
  entries to `Config.HiddenLayers`.

- [ ] **T-7B02** — Add `Pattern(block []HiddenLayerSpec[T], repeats uint) *NN[T]` builder
  method. Appends `block × repeats` entries. Validation: `repeats == 0` or `len(block) == 0`
  → error at Compile().

- [ ] **T-7B03** — Add `HiddenLayers(layers []HiddenLayerSpec[T]) *NN[T]` builder method.
  **Replaces** any previously added hidden layers (setter, not appender).

- [ ] **T-7B04** — Add Options-style constructors: `Repeat[T](count, size, act)`,
  `Pattern[T](block, repeats)`, `WithHiddenLayers[T](layers)` in `pkg/nn/options.go`.
  Options style appends (consistent with existing `WithHiddenLayer`).

## Track C — GoNN Developer Skills (tooling)

*Goal: Create `skills/gonn/SKILL.md` + examples + resources for AI-assisted GoNN development.*
*Source: [l2-gonn-skills.md](../specifications/l2-gonn-skills.md)*

- [ ] **T-7C01** — Create `skills/gonn/SKILL.md` with YAML frontmatter (name: gonn) and
  10 sections from spec §5.2: API styles, lifecycle, types, bulk constructors, activation↔loss
  matching, optimizer selection, AI-Meta, error handling, testing patterns, common mistakes.

- [ ] **T-7C02** — Create `skills/gonn/examples/`: `builder-xor.md` (Style A XOR),
  `options-mnist.md` (Style B placeholder), `deep-network.md` (100-layer via Repeat/Pattern).

- [ ] **T-7C03** — Create `skills/gonn/resources/`: `api-reference.md` (condensed public API),
  `conventions.md` (C25–C33 translated for AI agents without referencing RULES.md).

## Validation Tasks

- [ ] **T-7T01** — Scheduler + Deep Builder validation (after Track A + B):
  - `go test -race ./pkg/optimizer/...` — scheduler tests green; SaveState/LoadState round-trips.
  - `go test -race ./pkg/nn/...` — Repeat/Pattern/HiddenLayers tests green.
  - `go test -bench=. -benchmem ./pkg/nn/...` — 100-layer Compile() under 1ms.
  - Coverage `pkg/optimizer/scheduler*`: ≥80%.

- [ ] **T-7T02** — Skills validation (after Track C):
  - `skills/gonn/SKILL.md` exists with valid YAML frontmatter.
  - All example code blocks in `skills/gonn/examples/*.md` pass `go vet` check.
  - API reference in `skills/gonn/resources/api-reference.md` covers all exported symbols.

## Gate

- [ ] **T-7Z01** — Phase 7 gate:
  - `go build ./...` clean.
  - `go test -race ./...` all green.
  - Every non-example package ≥80% line coverage.
  - `skills/gonn/` directory structure matches spec §5.1.
  - 3 orphaned specs resolved (l1-lr-scheduling, l2-deep-builder, l2-gonn-skills).
