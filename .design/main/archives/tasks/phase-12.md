---
phase: 12
name: Meta-Learning Hooks
status: Done
subsystem: pkg/nn (new meta.go), pkg/utils (sentinels)
requires:
  - phase-11 (v0.10.0 tagged; pkg/nn integration points stable)
  - l1-meta-learning-hooks Stable v1.0.0
  - l2-meta-learning-impl Stable v0.1.0
provides:
  - pkg/nn/meta.go (ParamAccessor[T], ScalarParam[T], SliceParam[T], MetaLearner[T], FeatureFunc[T])
  - pkg/nn (WithMetaLearner option, MetaLearner config field, train.go integration hook)
  - pkg/utils (ErrMetaLearnerShape, ErrMetaLearnerRunning sentinels)
key_files:
  created:
    - pkg/nn/meta.go
    - pkg/nn/meta_test.go
  modified:
    - pkg/nn/config.go
    - pkg/nn/options.go
    - pkg/nn/compile.go
    - pkg/nn/train.go
    - pkg/utils/errors.go
    - CHANGELOG.md
patterns_established:
  - ParamAccessor[T] uniform interface for meta-learning param registration
  - MetaLearner advisory hook (Warn on error, does NOT abort training)
  - continue-on-error semantics in step() — apply all, collect first error
duration_minutes: 45
---

# Phase 12 — Meta-Learning Hooks

**Status:** Todo
**Decomposed:** 2026-05-17
**Tasks:** 3 feature + 1 validation + 1 gate = 5 total.
**Specs:** l1-meta-learning-hooks v1.0.0 (Stable), l2-meta-learning-impl v0.1.0 (Stable)
**Track order:** Single track A (sequential); T-12T01 after A; Gate T-12Z01.

## Track A — Meta-Learning Hooks (pkg/nn/meta.go)

*Goal: Implement ParamAccessor[T] + MetaLearner[T] from l2-meta-learning-impl.md §5.2–5.3.*
*Source: [l2-meta-learning-impl.md](../specifications/l2-meta-learning-impl.md)*

- [x] **T-12A01** — Create `pkg/nn/meta.go` with type definitions.
  `ParamAccessor[T utils.Float]` interface (`Get() []T`, `Set([]T) error`, `Name() string`).
  `ScalarParam[T]` struct (`ptr *T`, `name string`); `Get` returns `[]T{*ptr}`; `Set` validates `len(v) == 1` else `ErrMetaLearnerShape`.
  `SliceParam[T]` struct (`ptr *[]T`, `name string`); `Get` returns copy; `Set` validates `len(v) == len(*ptr)` else `ErrMetaLearnerShape`.
  `FeatureFunc[T]` type alias for `func(loss T, iter int, maxIter int) []T`; package-level `DefaultFeatureFunc[T]` returns `[]T{loss, T(iter)/T(maxIter)}`.
  `MetaLearner[T]` struct fields: `inner *NN[T]`, `params []ParamAccessor[T]`, `Features FeatureFunc[T]` (defaults to `DefaultFeatureFunc[T]` if nil).
  Add `ErrMetaLearnerShape` and `ErrMetaLearnerRunning` to `pkg/utils/errors.go` per §5.4.
  **Verify**: `go build ./pkg/nn/... ./pkg/utils/...` clean; `go vet ./pkg/nn/...` clean.

- [x] **T-12A02** — Implement `(m *MetaLearner[T]) step(loss T, iter, maxIter int) error`.
  Sequence (per l2-meta-learning-impl.md §5.3 diagram):
  1. If `m.Features == nil`, fall back to `DefaultFeatureFunc[T]`.
  2. Build feature vector: `features := m.Features(loss, iter, maxIter)`.
  3. Call `output := m.inner.Query(features)` (error → wrap and return).
  4. Validate `len(output) == len(m.params)`; mismatch → `ErrMetaLearnerShape` wrapped with `fmt.Errorf("meta-learner: inner output %d != %d params: %w", len(output), len(m.params), utils.ErrMetaLearnerShape)`.
  5. For each `i`, call `m.params[i].Set([]T{output[i]})`; collect FIRST error and return after applying all (continue-on-error semantics — partial application is intentional per §5.3 "continue others").
  **Verify**: TestMetaLearnerStepDefaultFeatures, TestMetaLearnerStepCustomFeatures, TestMetaLearnerStepShapeMismatch, TestMetaLearnerStepInnerQueryError — all green.

- [x] **T-12A03** — Wire `MetaLearner[T]` into `pkg/nn/`.
  `pkg/nn/config.go`: add `MetaLearner *MetaLearner[T]` field (nil = disabled).
  `pkg/nn/options.go`: add `WithMetaLearner[T utils.Float](ml *MetaLearner[T]) Option[T]`:
    - Validates outer NN `control.Load() == Idle`; else returns option that errors `ErrMetaLearnerRunning` at compile-time.
    - Sets `cfg.MetaLearner = ml`.
  `pkg/nn/train.go`: after `opt.Step(...)` call, before `fireEvent(OnIterationEnd, ...)`:
    ```go
    if nn.cfg.MetaLearner != nil {
        if err := nn.cfg.MetaLearner.step(loss, iter, int(nn.cfg.MaxIterations)); err != nil {
            // Log via slog (Warn level) — do NOT abort training per §5.3 (meta-step is advisory)
            nn.logger.Warn("meta-learner step failed", "error", err)
        }
    }
    ```
  No change to `Train()` signature or return type.
  **Verify**: TestWithMetaLearnerWiring (verify cfg field set), TestWithMetaLearnerRejectsRunning (state guard fires), TestTrainCallsMetaLearnerStep (mock MetaLearner counts invocations across N iterations).

## Validation Tasks

- [x] **T-12T01** — Meta-Learning end-to-end validation.
  - `go test -count=1 ./pkg/nn/... ./pkg/utils/...` → exit 0; all new meta tests green (race flag deferred to CI per Windows gcc absence).
  - Coverage `pkg/nn/` ≥ 80% (must not regress from current 81.5%).
  - **Convergence check**: 2→1 inner NN tunes outer learning rate on a synthetic loss-decay task; converge faster than static LR (compare epochs-to-loss-limit, 3-seed average). Skip on CI low-power runners with `t.Skip` if benchmark wall time > 30s.
  - `ErrMetaLearnerShape` fires when inner output size ≠ registered params (TestMetaLearnerStepShapeMismatch).
  - `ErrMetaLearnerRunning` fires when `WithMetaLearner` applied during training (TestWithMetaLearnerRejectsRunning).
  - JSON round-trip: NN with `MetaLearner` configured serializes/deserializes inner NN per `l1-network-persistence` PERS-2 — defer to T-12Z01 if outside scope of v0.11.0.

## Gate

- [x] **T-12Z01** — Phase 12 gate (v0.11.0 release):
  - `go build ./...` clean (zero regressions across all 19 packages + examples).
  - `go test -count=1 -race ./...` all green (CI must enforce `-race`; local Windows runs without per gcc absence).
  - Coverage floor: `pkg/nn/` ≥ 80% (meta additions); `pkg/utils/` unchanged.
  - `CHANGELOG.md` v0.11.0 entry written: meta-learning hooks shipped (ParamAccessor[T], MetaLearner[T], WithMetaLearner).
  - `v0.11.0` git tag created per `l1-release-policy.md §5.4` (user runs `git tag -a v0.11.0`).
  - If `examples/E11_meta_learning/` (E11 from catalog) authored — bundle in this release; else defer to Phase 13.
