---
phase: 16
name: "Recurrent Completion + GPU Backward + AI-Meta Rollout"
status: Todo
subsystem: "pkg/layer/recurrent/ (GRU + LastStep); pkg/optimizer/ (ClipByGlobalNorm); pkg/nn/ (recurrent options + WithBackend + WithGradClipNorm + compile wiring); pkg/compute/gpu/opencl/ (Dense Backward kernel + bench); cmd/lint-aimeta/ + pkg/aimeta/ (--resolve + RESOLVE rule); pkg/activation, pkg/loss, pkg/neuron, pkg/layer, pkg/network, pkg/dataset, pkg/checkpoint, pkg/compute, pkg/persistence, pkg/nn (AI-Meta annotations + TestAIMetaCompliance hooks rollout phases 3-5)"
requires:
  - "Phase 15 ✓ (v0.13.0 RC ready)"
  - "l1-recurrent-layers Stable v0.1.0 ✓ (REC-1..9 invariants)"
  - "l2-recurrent-impl Stable v0.1.0 ✓ (defers GRU/LastStep/grad-clip/nn-options to §6 phases γ-θ)"
  - "l2-backend-gpu Stable v0.1.0 ✓ (defers OpenCL Backward + nn fallback + perf gate to §6 phases C-F)"
  - "l2-aimeta-linter Stable v0.1.0 ✓ (defers --resolve to §5.5 phase C + rollout to §8 phases 3-5)"
  - "l2-ai-doc-metadata Stable v1.0.0 ✓ (parent vocabulary contract)"
  - "Phase 15 patterns_established: Orthogonal init helper, build-tag isolation for cgo backends, TestAIMetaCompliance hook template"
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 16 Tasks — Recurrent Completion + GPU Backward + AI-Meta Rollout

**Phase:** 16
**Status:** Todo
**Strategic Goal:** Close out Phase 15's three deferred tracks in a single release pass.
(A) Bring `pkg/layer/recurrent/` to full L2 spec coverage: GRU[T] cell, LastStep[T] sequence collapser,
`ClipByGlobalNorm[T]` optimizer helper + `WithGradClipNorm` option, and wire
`WithSimpleRNN`/`WithLSTM`/`WithGRU`/`WithLastStep` options through `pkg/nn/options.go` + `compile.go`.
(B) Land the OpenCL Dense Backward kernel (cross-referenced vs CPU within tolerance 1e-4), wire
`WithBackend(compute.Backend[T])` option into `pkg/nn/` with graceful `ErrBackendUnavailable` fallback
to CPU baseline, and gate Forward+Backward GPU vs CPU benchmark expectations.
(C) Add `--resolve` flag + RESOLVE rule code to `cmd/lint-aimeta/` (auto-fix known violations), then
roll `TestAIMetaCompliance` hooks across the remaining 10 packages (rollout phases 3+4+5 from
`l2-aimeta-linter.md §8`). CUDA mirror, dropout for attention, and full `pkg/nn` rollout polish
explicitly deferred to Phase 18+ per @role:planner audit below. Target release: v0.14.0.

## Atomic Checklist

### Track A — Recurrent Completion (parallel-safe: A01 ∥ A02 ∥ A03; A04 sequential after A01+A02+A03)

- [ ] [T-16A01] `pkg/layer/recurrent/gru.go` — `GRU[T]` 3-gate (reset, update, candidate) Forward + Backward + Init + Step + MarshalJSON/UnmarshalJSON
- [ ] [T-16A02] `pkg/layer/recurrent/laststep.go` — `LastStep[T]` stateless sequence→vector collapser (final timestep selection)
- [ ] [T-16A03] `pkg/optimizer/clip.go` + `pkg/nn/options.go` — `ClipByGlobalNorm[T](threshold T)` + `WithGradClipNorm[T](threshold T)` option + train.go pre-step wiring
- [ ] [T-16A04] `pkg/nn/options.go` + `pkg/nn/compile.go` — `WithSimpleRNN[T]`/`WithLSTM[T]`/`WithGRU[T]`/`WithLastStep[T]` options + recurrent stack prepend wiring + `setupRecurrentShapes` time-major pre-pass

### Track B — GPU Backward + Fallback Wiring (B01 first; B02 ∥ B03 after B01)

- [ ] [T-16B01] `pkg/compute/gpu/opencl/kernels.cl` + `kernels.go` — Dense Backward kernel pair (weight gradient + input gradient); cross-reference vs `pkg/compute/cpu/` within tolerance 1e-4
- [ ] [T-16B02] `pkg/nn/options.go` + `pkg/nn/compile.go` — `WithBackend(compute.Backend[T])` option; `ErrBackendUnavailable` graceful fallback to CPU baseline (no-op for existing examples)
- [ ] [T-16B03] `pkg/compute/gpu/opencl/bench_test.go` — Dense Forward+Backward GPU vs CPU performance bench (`-bench=.` opt-in); document speedup floor (≥2× for ≥256×256 matrices on supported hardware)

### Track C — AI-Meta Rollout + --resolve Flag (C01 first; C02 ∥ C03 after C01)

- [ ] [T-16C01] `pkg/aimeta/resolver.go` + `cmd/lint-aimeta/main.go` + `cmd/lint-aimeta/resolve.go` — `--resolve` flag wiring + RESOLVE rule code + known-recipe fixes (missing indent, lowercase labels, missing terminal newline)
- [ ] [T-16C02] Rollout phase 3+4 — `pkg/activation/aimeta_test.go`, `pkg/loss/aimeta_test.go`, `pkg/neuron/aimeta_test.go`, `pkg/layer/aimeta_test.go` (norm+conv siblings reuse same hook), `pkg/network/aimeta_test.go` + AI-Meta annotations on each package's exported symbols
- [ ] [T-16C03] Rollout phase 5 — `pkg/dataset/aimeta_test.go`, `pkg/checkpoint/aimeta_test.go`, `pkg/compute/aimeta_test.go`, `pkg/persistence/aimeta_test.go`, `pkg/nn/aimeta_test.go` + AI-Meta annotations on each package's exported symbols

### Validation

- [ ] [T-16T01] Validation Track A — finite-difference gradient check on `GRU[T]` Forward/Backward (4 cases incl. 1-step + multi-step); `LastStep[T]` shape round-trip; `ClipByGlobalNorm` invariant (||g||₂ ≤ threshold after clip); `pkg/layer/recurrent/` coverage ≥80%
- [ ] [T-16T02] Validation Track B — GPU Dense Backward correctness vs CPU baseline (random weights + inputs, 5 trials, tolerance 1e-4); fallback test (no `opencl` build tag → CPU path returns same result); `pkg/compute/gpu/` coverage ≥80%
- [ ] [T-16T03] Validation Track C — `--resolve` round-trip on golden testdata/violating_pkg/ (resolve → re-lint → 0 violations); full rollout passes `go test ./pkg/...` with new `aimeta_test.go` in every rolled-out package

### Gate

- [ ] [T-16Z01] Phase 16 release gate — `go build ./...` clean (default tags, no cgo); each new/touched package individually ≥80% coverage; `pkg/nn/` ≥75% (compile.go growth offset); CHANGELOG.md v0.14.0 entry; `v0.14.0` tag prepared (`git tag -a` left to user per finalization protocol)

## Detailed Tracking

### [T-16A01] `pkg/layer/recurrent/gru.go`

- **Spec:** `l2-recurrent-impl.md` §6 phase γ + REC-3 (state update math)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestGRU -count=1 ./pkg/layer/recurrent/` PASS; finite-difference grad check max_abs_err < 1e-4 (T-16T01); coverage line for `gru.go` ≥85%
- **Handoff:** A04 wires `WithGRU[T]` option after this lands.
- **Notes:** Implement 3-gate (reset r_t, update z_t, candidate ñ_t) per GRU canonical equations. Reuse `cell.go` sigmoid/tanh fused helpers from Phase 15. Forget-bias init not applicable (GRU has no forget gate). Use Xavier for W_x + Orthogonal for W_h, same as LSTM.

### [T-16A02] `pkg/layer/recurrent/laststep.go`

- **Spec:** `l2-recurrent-impl.md` §6 phase δ + REC-8 (sequence-to-vector reduction)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestLastStep -count=1 ./pkg/layer/recurrent/` PASS; Forward picks final-timestep slice; Backward scatters gradient into final-timestep position with zeros elsewhere; coverage line for `laststep.go` ≥85%
- **Handoff:** A04 wires `WithLastStep[T]` option after this lands.
- **Notes:** Stateless reshape — `[T, B, H]` → `[B, H]`. No parameters. Implements `layer.Layer[T]` via Forward/Backward only (Init/Step return nil, JSON round-trip is identity).

### [T-16A03] `pkg/optimizer/clip.go` + `pkg/nn/options.go`

- **Spec:** `l2-recurrent-impl.md` §6 phase ε + REC-7 (BPTT stability) + l1-recurrent-layers REC-7 invariant
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestClipByGlobalNorm -count=1 ./pkg/optimizer/` PASS; invariant ||g||₂ ≤ threshold after clip across all parameter slices; `go test -run TestWithGradClipNorm -count=1 ./pkg/nn/` PASS; pkg/optimizer coverage ≥85%
- **Handoff:** A04 wires recurrent options that benefit from clip; train.go calls `clip.Apply(gradients)` before `opt.Step` when `cfg.GradClipNorm > 0`.
- **Notes:** Compute global L2 norm across all parameter gradient slices, scale by `min(1, threshold/||g||₂)`. Stdlib-only (`math.Sqrt`). Wire as `WithGradClipNorm[T](threshold T) Option[T]` setting `cfg.GradClipNorm`. train.go check: `if cfg.GradClipNorm > 0 { clip.ApplyByGlobalNorm[T](grads, cfg.GradClipNorm) }`.

### [T-16A04] `pkg/nn/options.go` + `pkg/nn/compile.go` (recurrent options + setupRecurrentShapes)

- **Spec:** `l2-recurrent-impl.md` §6 phases ζ-θ + REC-9 (composition rules)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestRecurrentCompile -count=1 ./pkg/nn/` PASS; `WithSimpleRNN`/`WithLSTM`/`WithGRU`/`WithLastStep` options compose with Dense/Output; recurrent stack prepended before Dense like Conv2D prefix; shape inference handles `[T, B, H_in] → [T, B, H_out] → ... → [B, H_final]` (LastStep) → Dense
- **Handoff:** B02 follows on same compile.go file; sequence: A04 commits → B02 rebases.
- **Notes:** Mirror Conv2D prefix pattern from Phase 14. Add `RecurrentPrefix []layer.Layer[T]` field to config. `setupRecurrentShapes` does time-major pre-pass to validate `[T, B, H]` shapes through the chain. Reuse `WithInputShape` for initial `[T, B, H_in]` declaration. Sequential dependency: A01+A02+A03 must land before A04 (A04 imports them).

### [T-16B01] `pkg/compute/gpu/opencl/kernels.cl` + `kernels.go` (Dense Backward)

- **Spec:** `l2-backend-gpu.md` §6 phase C + COMP-3 invariant (Backward parity vs CPU)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -tags='cgo opencl' -run TestDenseBackward -count=1 ./pkg/compute/gpu/opencl/` PASS on machine with OpenCL runtime; `cgo,opencl`-absent build tag → test skipped (not failed); max_abs_err < 1e-4 (float32) vs `pkg/compute/cpu/` baseline
- **Handoff:** B02 + B03 depend on this kernel being callable.
- **Notes:** Two OpenCL kernels — `dense_grad_w` (∂L/∂W = X^T·∂L/∂Y) and `dense_grad_x` (∂L/∂X = ∂L/∂Y·W^T). Same build-tag isolation pattern from Phase 15 (`//go:build cgo && opencl`). Add a 5-trial randomized fixture in `opencl_test.go` comparing against `pkg/compute/cpu/dense.go`.

### [T-16B02] `pkg/nn/options.go` + `pkg/nn/compile.go` (WithBackend + ErrBackendUnavailable fallback)

- **Spec:** `l2-backend-gpu.md` §6 phase D + l1-compute-backend §5.3 (graceful fallback)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestWithBackend -count=1 ./pkg/nn/` PASS; with `compute.CPUBackend[T]()` (default) — existing examples unaffected; with `gpu.NewOpenCL[T]()` (build tag absent) — `ErrBackendUnavailable` returned, compile() falls back to CPU and logs a Warn via `pkg/utils/logger.go`; `go build ./...` clean default tags
- **Handoff:** B03 benchmark uses this wiring; gate T-16Z01 checks compile().
- **Notes:** Add `Backend compute.Backend[T]` field to config. Default: `compute.CPUBackend[T]()`. Option signature: `WithBackend[T](b compute.Backend[T]) Option[T]`. Fallback policy: `errors.Is(err, utils.ErrBackendUnavailable)` → assign CPU baseline + emit `logger.Warn("backend unavailable, falling back to CPU", "err", err)`. Bench in B03 separate.

### [T-16B03] `pkg/compute/gpu/opencl/bench_test.go`

- **Spec:** `l2-backend-gpu.md` §6 phase E + PERF-5 invariant (benchmark documentation)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -tags='cgo opencl' -bench=BenchmarkDense -benchmem -count=3 ./pkg/compute/gpu/opencl/` runs; document baseline ns/op vs CPU for 64×64, 256×256, 1024×1024 matrices in package doc; gate only requires bench compiles + runs, not specific speedup numbers (hardware-dependent)
- **Handoff:** Closes Track B.
- **Notes:** Forward + Backward + paired Forward-Backward benchmarks. Use `b.ResetTimer()` after buffer allocation. Document expected speedup floor in comment block above `BenchmarkDenseForward256` — "≥2× for ≥256×256 on Intel/AMD iGPU; ≥5× on discrete NVIDIA/AMD". CI runs without `opencl` tag — bench file body guarded by build tag.

### [T-16C01] `pkg/aimeta/resolver.go` + `cmd/lint-aimeta/{main.go,resolve.go}`

- **Spec:** `l2-aimeta-linter.md` §5.5 phase C + RESOLVE rule code
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestResolver -count=1 ./pkg/aimeta/` PASS; `go test -run TestResolve -count=1 ./cmd/lint-aimeta/` PASS; resolve → re-lint round-trip on `testdata/violating_pkg/` yields 0 violations (T-16T03); coverage for new resolver.go ≥80%
- **Handoff:** Track C parallel rollout (C02 + C03) uses `--resolve` to bootstrap AI-Meta annotations on newly-rolled packages.
- **Notes:** Resolver applies fixes for INDENT (`/AI-` → `//AI-`? no, AI-Meta prefix is `/`), LABEL (lowercase → Title-Case), LAST (missing terminal newline). Skip VOCAB/TIER/MULTI/ENUM/ARTIFACT — those require human design judgement. New rule code RESOLVE-1..N enumerated in `pkg/aimeta/violation.go` per spec §5.5.

### [T-16C02] Rollout phase 3+4 — `pkg/activation` + `pkg/loss` + `pkg/neuron` + `pkg/layer` + `pkg/network`

- **Spec:** `l2-aimeta-linter.md` §8 rollout phase 3 + phase 4
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestAIMetaCompliance -count=1 ./pkg/activation/ ./pkg/loss/ ./pkg/neuron/... ./pkg/layer/... ./pkg/network/` all PASS; each package has `aimeta_test.go` calling `aimeta.Check(".", aimeta.DefaultOptions())`; AI-Meta annotations attached to each exported symbol per `l2-ai-doc-metadata.md §4`
- **Handoff:** C03 follows; both can run in parallel after C01 (--resolve) closes.
- **Notes:** Use `cmd/lint-aimeta --resolve ./pkg/<dir>/...` (from C01) to bootstrap missing annotations, then hand-edit ambiguous cases. `pkg/layer/` covers `norm/` + `conv/` + `recurrent/` siblings via same hook (single `aimeta_test.go` at `pkg/layer/aimeta_test.go` checking all subdirs). Existing AI-Meta annotations from Phase 6+ work in `pkg/optimizer/` and `pkg/regularizer/` should already pass — verify but not retouch.

### [T-16C03] Rollout phase 5 — `pkg/dataset` + `pkg/checkpoint` + `pkg/compute` + `pkg/persistence` + `pkg/nn`

- **Spec:** `l2-aimeta-linter.md` §8 rollout phase 5 (top-level packages, last set)
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `go test -run TestAIMetaCompliance -count=1 ./pkg/dataset/ ./pkg/checkpoint/ ./pkg/compute/... ./pkg/persistence/ ./pkg/nn/` all PASS; each package has `aimeta_test.go`; `pkg/nn` annotations cover the largest API surface (Options, Builder, Fit/Train/Query) — highest scrutiny
- **Handoff:** Closes Track C; feeds Gate T-16Z01.
- **Notes:** `pkg/nn` is the largest exposed API surface. Allocate dedicated review pass for Options (`WithXxx` family — ~30 functions). `pkg/compute/...` covers both `cpu/` and `gpu/` subdirs (single `aimeta_test.go` at `pkg/compute/aimeta_test.go`). Use `cmd/lint-aimeta --resolve` for mechanical fixes, hand-edit semantic content (Purpose/Behavior labels).

### [T-16T01] Validation Task — Track A

- **Goal:** Verify GRU forward+backward gradient correctness, LastStep shape round-trip, GradClipNorm invariant, all against spec REC-3/REC-7/REC-8 + Phase 15's existing finite-difference pattern.
- **Method:**
  - `go test -run TestGRUGradientCheck -count=1 ./pkg/layer/recurrent/` (4 cases: 1-step, 3-step, 5-step, mismatched H_in/H_out)
  - `go test -run TestLastStepRoundTrip -count=1 ./pkg/layer/recurrent/`
  - `go test -run TestClipByGlobalNormInvariant -count=1 ./pkg/optimizer/` (5 random ||g|| values incl. ||g|| < threshold no-op)
  - `go test -cover ./pkg/layer/recurrent/` ≥80%; `go test -cover ./pkg/optimizer/` ≥85% (already 88%+)
- **Status:** Todo

### [T-16T02] Validation Task — Track B

- **Goal:** Verify GPU Dense Backward kernel matches CPU baseline within tolerance; fallback test for missing opencl runtime.
- **Method:**
  - `go test -tags='cgo opencl' -run TestDenseBackwardParity -count=1 ./pkg/compute/gpu/opencl/` (5 random trials, max_abs_err < 1e-4)
  - `go test -run TestBackendFallback -count=1 ./pkg/nn/` (default tags — `gpu.NewOpenCL[float64]()` returns ErrBackendUnavailable → CPU fallback active)
  - `go test -cover ./pkg/compute/gpu/...` ≥80%
- **Status:** Todo

### [T-16T03] Validation Task — Track C

- **Goal:** Verify --resolve correctness on golden testdata + every rolled-out package passes TestAIMetaCompliance.
- **Method:**
  - `go test -run TestResolveRoundTrip -count=1 ./cmd/lint-aimeta/` (resolve testdata/violating_pkg → re-lint → 0 violations)
  - `go test -run TestAIMetaCompliance ./pkg/...` (all 13 packages: activation, loss, neuron, layer, network, dataset, checkpoint, compute, persistence, nn, optimizer, regularizer, utils — utils already done in Phase 15)
- **Status:** Todo

### [T-16Z01] Phase 16 Release Gate

- **Goal:** Verify all three tracks merge cleanly and the v0.14.0 release candidate is healthy.
- **Method:**
  - `go build ./...` clean (default tags, no cgo)
  - `go test ./...` green (per-package on Windows VA-constrained host; full `./...` in CI)
  - Coverage gate: `pkg/layer/recurrent/` ≥80%, `pkg/optimizer/` ≥85%, `pkg/compute/gpu/...` ≥80%, `pkg/aimeta/` ≥80%, `cmd/lint-aimeta/` ≥80%, all newly-rolled aimeta_test.go packages PASS
  - CHANGELOG.md has v0.14.0 entry with three track summaries
  - `git tag -a v0.14.0` left to user per finalization protocol (no auto-commit)
- **Status:** Todo

## @role:planner Audit

**Optimism Bias** — Phase 15 audit deferred 9 items to "Phase 16+". This phase claims 8 of those 9 (deferring only CUDA mirror to Phase 18+ — CUDA replication of Track B would double the workload). Track A reduced from full spec's §6 phases α-θ (Phase 15 closed α+β) to just γ+δ+ε+ζ-θ collapsed into 4 tasks. Track C rollout phases 2-5 (4 phases × ~3 pkgs each = 12 packages × hook) collapsed to 2 tasks (C02+C03) by grouping. Risk: C02+C03 are LOC-heavy mechanical changes (AI-Meta annotations on ~150 exported symbols). Mitigation: `--resolve` from C01 bootstraps mechanical fixes; remaining hand-edit is semantic labelling only.

**Hidden Dependencies** — Track A's A04 and Track B's B02 both modify `pkg/nn/options.go` + `pkg/nn/compile.go`. Sequential ordering: A04 must commit before B02 (or merge with conflict resolution). Track C's C02+C03 modify `pkg/aimeta/violation.go` for RESOLVE rule (already done in C01). No cross-track resource overlap beyond the A04/B02 pkg/nn pair. Track A's A03 (clip.go) adds to `pkg/optimizer/` — independent of A01/A02 (`pkg/layer/recurrent/`).

**Cascade Risk** — B02 (`compile.go` backend fallback wiring) is the highest-risk single task: if it breaks compile(), all 19 packages with examples fail. Mitigation: default to `compute.CPUBackend[T]()` (no-op for existing examples), GPU as opt-in via `WithBackend()`. A04 (recurrent options + compile wiring) is medium risk — mitigated by Phase 14's Conv2D prefix pattern as proven template. C03's `pkg/nn` annotation set is large (~50 symbols) — mitigated by `--resolve` mechanical pass + dedicated review.

## Deferred to Phase 18+

- CUDA mirror of OpenCL bindings (Track B parity for NVIDIA GPUs) — `pkg/compute/gpu/cuda/` per `l2-backend-gpu.md §3.2`
- Recurrent layer Dropout variant — pending RECURRENT-DROPOUT spec amendment per `l2-recurrent-impl.md` future-work note
- `pkg/aimeta` performance benchmark gate — currently AST walk is O(files × symbols); spec §9 marks as nice-to-have
