---
phase: 10
name: Normalization Layers + Training Callbacks
status: Todo
subsystem: pkg/layer/norm (new), pkg/nn (callbacks + options), pkg/utils
requires:
  - phase-9 (v0.8.0 tagged; pkg/network topology stable; pkg/nn options/compile/train stable)
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 10 — Normalization Layers + Training Callbacks

**Status:** Todo
**Decomposed:** 2026-05-11
**Tasks:** 7 feature + 2 validation + 1 gate = 10 total
**Specs:** l2-normalization-impl v0.1.0, l2-callbacks-impl v0.1.0
**Track order:** A and B parallel for package code; coordinate on `pkg/nn/train.go`
               (Track B modifies train.go, Track A adds propagateMode call — merge after B);
               T-10T01 after Track A, T-10T02 after Track B; Gate T-10Z01 after all.

## Track A — Normalization Layers (pkg/layer/norm/)

*Goal: Implement Normalizer[T] interface + BatchNorm/LayerNorm/GroupNorm from
l2-normalization-impl.md. New sub-package in pkg/layer/norm/.*
*Source: [l2-normalization-impl.md](../specifications/l2-normalization-impl.md)*

- [ ] **T-10A01** — Create `pkg/layer/norm/norm.go`: `NormMode` enum (`NormTrain=0`, `NormEval=1`);
  `Normalizer[T utils.Float]` interface (`Forward`, `SetMode`, `GradSlots`, `InputSize`, `OutputSize`);
  shared free functions: `stddev[T](variance, eps T) T`, `applyAffine[T](xHat, gamma, beta []T) []T`.
  No state — interface + pure helpers only. Add compile-time assertions for all three concrete types.

- [ ] **T-10A02** — Create `pkg/layer/norm/batchnorm.go` + begin `norm_test.go`:
  `BatchNorm[T]`: fields `features int`, `eps T`, `momentum T`, `affine bool`, `gamma, beta []T`,
  `runningMean, runningVar []T`, `mode atomic.Int32`.
  `NewBatchNorm[T](features int, opts ...BatchNormOption[T]) *BatchNorm[T]` — γ=1, β=0, running stats 0/1.
  `Forward(x []T) []T`: train path uses batch mean/var + EMA update; eval path uses running stats.
  `SetMode`, `GradSlots`, `InputSize/OutputSize`, `MarshalJSON/UnmarshalJSON`.
  `ErrBatchNormSingleSample` guard in train mode.
  Tests: shape preservation, γ=1/β=0 identity check, EMA update, eval uses frozen stats, JSON round-trip.

- [ ] **T-10A03** — Create `pkg/layer/norm/layernorm.go` and `pkg/layer/norm/groupnorm.go`.
  `LayerNorm[T]`: per-sample mean/var; no running stats; affine optional. `MarshalJSON/UnmarshalJSON`.
  `GroupNorm[T]`: G groups; `ErrGroupSizeMismatch` if `features % G != 0`. `MarshalJSON/UnmarshalJSON`.
  Extend `norm_test.go`: LayerNorm shape + affine disabled; GroupNorm group partition + G-divides-F guard;
  SetMode has no effect on LayerNorm/GroupNorm (forward is always per-sample).

- [ ] **T-10A04** — Wire normalization into `pkg/nn/`:
  `pkg/nn/options.go`: add `WithNormAfterLayer[T](idx int, n layer.Normalizer[T]) Option[T]`,
  `WithBatchNorm[T](idx int) Option[T]`, `WithLayerNorm[T](idx int) Option[T]`.
  `pkg/nn/config.go`: add `NormLayers map[int]layer.Normalizer[T]` field.
  `pkg/nn/compile.go`: inject norm layers into the compiled hidden stack; wire `GradSlots()` into
  the optimizer step alongside Dense weight gradients.
  `pkg/nn/nn.go`: add `SetTrain()`/`SetEval()` methods that propagate `NormTrain`/`NormEval` to all
  registered `Normalizer[T]` instances. **Coordinate with Track B before modifying train.go.**

## Track B — Training Callbacks (pkg/nn/callbacks.go)

*Goal: Implement CallbackRegistry[T] + train.go integration from l2-callbacks-impl.md.*
*Source: [l2-callbacks-impl.md](../specifications/l2-callbacks-impl.md)*

- [ ] **T-10B01** — Create `pkg/nn/callbacks.go`:
  `StopReason` enum (6 values per spec §5.2); `ErrStopTraining` sentinel;
  `CallbackContext[T]` struct (Iteration, Loss, MinLoss, MinIter, StopReason *StopReason, Snapshot);
  `CallbackFn[T]` type alias; `CallbackRegistry[T]` struct (3 slice fields).
  `invokeOne[T](fn, ctx)` — wraps in `recover()`, emits `utils.Errorf(ErrCallbackPanic, ...)` on panic,
  returns nil (training continues). `fireEvent[T](fns, ctx)` — nil-check short-circuit (CB-3);
  iterate in order; first `errors.Is(err, ErrStopTraining)` → return stop signal.
  Add `ErrCallbackPanic` sentinel to `pkg/utils/errors.go`.

- [ ] **T-10B02** — Integrate callbacks into `pkg/nn/train.go`:
  Add `defer fireOnTrainEnd(nn.callbacks, &stopReason, &result)` at top of `Train()`.
  After `opt.Step` + reg hooks: if `loss < minLoss` → `fireEvent(OnImprovementFound, ctx)` first;
  then `fireEvent(OnIterationEnd, ctx)`. On stop signal from either → goto rollback+return path.
  `fireOnTrainEnd` builds `CallbackContext` with `StopReason` set and fires `OnTrainEnd` slice.
  **No change to `Train()` signature or return type.**
  After T-10B02: Add `propagateMode()` call from T-10A04 to `pkg/nn/nn.go` (coordinate merge).

- [ ] **T-10B03** — Wire callback options into `pkg/nn/`:
  `pkg/nn/options.go`: add `WithOnIterationEnd[T]`, `WithOnImprovementFound[T]`, `WithOnTrainEnd[T]`
  functional options (each appends to respective registry slice on the NN config).
  `pkg/nn/config.go`: add `Callbacks *CallbackRegistry[T]` field; lazy-init in `WithOn*` options.
  Create `pkg/nn/callbacks_test.go`: registration order, StopTraining from OnImprovementFound,
  panic recovery continues training, OnTrainEnd fires on normal+stop+panic exit,
  `BenchmarkNoCallbacks` → 0 allocs/op.

## Validation Tasks

- [ ] **T-10T01** — Normalization layers validation (after Track A):
  - `go test -count=1 -race ./pkg/layer/norm/...` — all tests green.
  - Coverage `pkg/layer/norm/` ≥80 %.
  - Verify `WithBatchNorm(0)` on a compiled NN does not change output shape.
  - Verify `SetEval()` freezes BatchNorm running stats (2nd forward pass == 1st eval pass).
  - Verify JSON round-trip: save NN with norm layer, reload, forward output bit-identical.

- [ ] **T-10T02** — Training callbacks validation (after Track B):
  - `go test -count=1 -race ./pkg/nn/...` — all tests green (including new callbacks_test.go).
  - `BenchmarkNoCallbacks` reports 0 allocs/op.
  - Verify `ErrStopTraining` from `OnImprovementFound` triggers min-loss rollback (TRN-3 preserved).
  - Verify `OnTrainEnd` fires when `Train()` returns normally AND when StopTraining is signalled.
  - Verify panicking callback does not abort training (CB-5): remaining iterations continue.

## Gate

- [ ] **T-10Z01** — Phase 10 gate:
  - `go build ./...` clean (zero regressions).
  - `go test -count=1 -race ./...` all green.
  - Every new / modified package ≥80 % line coverage:
    `pkg/layer/norm/` (new), `pkg/nn/` (callbacks addition), `pkg/utils/` (new sentinel).
  - `CHANGELOG.md` v0.9.0 entry written (normalization layers + training callbacks).
  - `v0.9.0` git tag created per `l1-release-policy.md §5.4`.
