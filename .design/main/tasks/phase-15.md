---
phase: 15
name: "Recurrent Foundation + GPU Backend Skeleton + AI-Meta Linter"
status: In Progress
subsystem: "pkg/layer/recurrent/ (new SimpleRNN + LSTM); pkg/compute/gpu/ (new umbrella + opencl skeleton); pkg/aimeta/ (new); cmd/lint-aimeta/ (new)"
requires:
  - "Phase 14 ✓ (v0.12.0 RC)"
  - "l1-recurrent-layers Stable v0.1.0 ✓"
  - "l2-recurrent-impl Stable v0.1.0 ✓"
  - "l2-backend-gpu Stable v0.1.0 ✓"
  - "l2-aimeta-linter Stable v0.1.0 ✓"
  - "l2-ai-doc-metadata Stable v1.0.0 ✓ (parent contract for Track C)"
  - "l1-compute-backend Stable v1.0.0 ✓ (parent for Track B)"
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 15 Tasks — Recurrent Foundation + GPU Backend Skeleton + AI-Meta Linter

**Phase:** 15
**Status:** Todo
**Strategic Goal:** Land foundation scope across three independent feature areas in parallel: (A) Recurrent layers `SimpleRNN[T]` + `LSTM[T]` end-to-end including BPTT + orthogonal init; (B) GPU compute backend umbrella package + OpenCL skeleton with Dense Forward kernel; (C) AI-Meta linter `pkg/aimeta` grammar package + `cmd/lint-aimeta` CLI + first per-package compliance hook. Secondary scope (GRU, LastStep, GradClipNorm, CUDA, perf benchmark gate, resolver `--resolve` flag, full rollout of `TestAIMetaCompliance` across packages) is explicitly deferred to Phase 16+. Target release: v0.13.0.

## Atomic Checklist

### Track A — Recurrent Foundation (parallel-safe: A01/A02 sequential; A03/A04 parallel-after-A02)

- [ ] [T-15A01] `pkg/utils/init.go` — `Orthogonal[T](rng, n)` helper via modified Gram-Schmidt
- [ ] [T-15A02] `pkg/layer/recurrent/cell.go` + `doc.go` — shared helpers (sigmoid/tanh fused, state cache buffers, layer-interface boilerplate)
- [ ] [T-15A03] `pkg/layer/recurrent/simple_rnn.go` — `SimpleRNN[T]` Forward + Backward + Init + Step + MarshalJSON/UnmarshalJSON
- [ ] [T-15A04] `pkg/layer/recurrent/lstm.go` — `LSTM[T]` 4-gate fused-matmul Forward + Backward + Init + Step + MarshalJSON/UnmarshalJSON

### Track B — GPU Backend Skeleton (parallel with Track A; B01 sequential before B02)

- [ ] [T-15B01] `pkg/compute/gpu/` umbrella — `doc.go` constants (`VendorOpenCL`/`VendorCUDA`), `unavailable.go` shim returning `ErrBackendUnavailable`, sentinel additions to `pkg/utils/errors.go` (`ErrBackendUnavailable`/`ErrBackendTransfer`/`ErrBackendKernel`)
- [ ] [T-15B02] `pkg/compute/gpu/opencl/` skeleton — `opencl.go` cgo bindings + `clGetPlatformIDs` discovery, `buffer.go` `Buffer[T]` wrapping `cl_mem`, `Allocate`/`Free`/`Write`/`Read` with round-trip test (no kernels yet)
- [ ] [T-15B03] `pkg/compute/gpu/opencl/kernels.go` + `kernels.cl` — Dense Forward kernel (matmul + bias + activation fused) launched via `clEnqueueNDRangeKernel`; cross-reference vs CPU within `cpu.ToleranceF32`/`ToleranceF64`

### Track C — AI-Meta Linter (parallel with Tracks A and B; C01 sequential before C02 before C03)

- [x] [T-15C01] `pkg/aimeta/` grammar package — `vocab.go` (closed `AllowedFields` + `Stability`/`Concurrency` enums + `TierRequiredFields`), `grammar.go` (`ParseBlock` returning `(Block, []Violation)`), `ast.go` (`ExtractBlock(*ast.CommentGroup)`), `violation.go` (`Violation` + 9 rule codes: LABEL/INDENT/CAP/LAST/VOCAB/TIER/MULTI/ENUM/ARTIFACT); golden-file tests in `pkg/aimeta/testdata/`
- [x] [T-15C02] `cmd/lint-aimeta/` CLI binary — `main.go` (dispatch to `aimeta.Check`), `output.go` (text + JSON formatters), `exitcode.go` (0/1/2/3 contract per l2-cli-client §5.3); end-to-end smoke test against `pkg/aimeta/testdata/` fixtures
- [x] [T-15C03] `pkg/utils/aimeta_test.go` — first per-package `TestAIMetaCompliance` hook (smallest package, simplest annotations); becomes the template for §8 rollout phases 2-5

### Validation + Gate

- [ ] [T-15T01] `pkg/layer/recurrent/{simple_rnn,lstm}_test.go` — finite-difference gradient check for `SimpleRNN` (3 cases) + `LSTM` (3 cases) within `1e-4` (float64) tolerance; orthogonality test for `utils.Orthogonal` (`‖Q·Q^T - I‖_F < 1e-10`)
- [x] [T-15T02] `pkg/aimeta/testdata/` golden-file matrix — all 5 §6 anti-patterns from `l2-ai-doc-metadata.md` produce exact expected `Violation` slices via `pkg/aimeta.Check`; clean fixtures produce empty violation list
- [ ] [T-15T03] `pkg/compute/gpu/unavailable_test.go` (pure-Go) — `gpu.New[T]("opencl")` without `cgo,opencl` build tag returns `ErrBackendUnavailable`; CPU fallback wired in `pkg/nn/compile.go` returns `cpu` backend on the same input with a `Warn` log captured by `slogtest`
- [ ] [T-15Z01] Phase 15 gate — `go build ./...` clean (default tags); `go build -tags 'opencl' ./...` clean on CI runner with OpenCL ICD installed (SKIP locally if unavailable); `go test ./...` all green; `pkg/layer/recurrent/` ≥80% coverage; `pkg/aimeta/` ≥80% coverage; `pkg/compute/gpu/` ≥80% coverage (excluding cgo-tagged files); v0.13.0 RC ready

## Detailed Tracking

### [T-15A01] `pkg/utils/init.go` — `Orthogonal[T](rng, n)` helper

- **Spec:** [l2-recurrent-impl.md](../specifications/l2-recurrent-impl.md) §5.8; [l1-recurrent-layers.md](../specifications/l1-recurrent-layers.md) REC-7
- **Method:** Modified Gram-Schmidt on N×N Gaussian matrix. Generate `A ∈ R^{n×n}` with `rng.NormFloat64()` entries (cast to T). For each column j: subtract projections onto columns 0..j-1, normalise to unit length. Return flattened row-major `[]T` of length `n*n`. Stdlib only.
- **Status:** Todo
- **Assignment:** Agent
- **Verify:**
  - `go test ./pkg/utils -run TestOrthogonal -v` — for `n ∈ {4, 32, 128}` and `T ∈ {float32, float64}`, compute `Q · Q^T` and assert Frobenius distance from identity `< 1e-10` (float64) / `< 1e-5` (float32).
  - `pkg/utils/` coverage ≥80%.
- **Handoff:** Triggers T-15A02 (cell.go uses `utils.Orthogonal` for `W_hh` init).

### [T-15A02] `pkg/layer/recurrent/cell.go` + `doc.go` — shared helpers

- **Spec:** [l2-recurrent-impl.md](../specifications/l2-recurrent-impl.md) §5.1 + §5.2 (boilerplate fields); [l2-ai-doc-metadata.md](../specifications/l2-ai-doc-metadata.md) (doc.go AI-Meta block)
- **Method:** `pkg/layer/recurrent/doc.go` — package doc + AI-Meta block per l2-ai-doc-metadata §5. `pkg/layer/recurrent/cell.go` — internal helpers: `applySigmoidFused(gates []T, offset, count int)`, `applyTanhFused(gates []T, offset, count int)`, `initRecurrentCache(seqLen, hidden int) (h, c []T)` (returns zeroed `[]T` of correct size; `c` is nil for non-LSTM callers). All helpers package-private.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-15A01 (cell.go does not call Orthogonal directly but verifies its presence in compile-time `var _ = utils.Orthogonal[float64]`).
- **Verify:**
  - `go build ./pkg/layer/recurrent/` clean.
  - `go test ./pkg/layer/recurrent -run TestCellHelpers -v` — table-driven sigmoid/tanh fused application matches `activation.Sigmoid`/`activation.TanH` element-wise within `1e-12`.
- **Handoff:** Triggers T-15A03 and T-15A04 (both cell types use the shared helpers).

### [T-15A03] `pkg/layer/recurrent/simple_rnn.go` — `SimpleRNN[T]`

- **Spec:** [l1-recurrent-layers.md](../specifications/l1-recurrent-layers.md) REC-2, REC-5, REC-6, REC-7, REC-8, REC-9; [l2-recurrent-impl.md](../specifications/l2-recurrent-impl.md) §5.2
- **Method:** Struct fields per §5.2 verbatim. `NewSimpleRNN[T](seqLen, inSize, hidden int)` constructs with zero-init weight slices. `Init(rng)` calls `utils.Xavier[T](rng, inSize)` for `Wxh`, `utils.Orthogonal[T](rng, hidden)` for `Whh`, zero for `Bh`. `Forward(input []T) []T` — iterate `t = 0..SeqLen-1`, compute `h_t = tanh(Wxh·x_t + Whh·h_{t-1} + b_h)`, cache `lastHidden[t+1] = h_t`. `Backward(upstream []T) []T` — walk `t = SeqLen-1 → 0`, accumulate `gradWxh`/`gradWhh`/`gradBh` per REC-5; return `gradX`. `Step(x)` and `ResetState()` per §5.10 pattern. `MarshalJSON`/`UnmarshalJSON` serialise `{Type:"SimpleRNN", SeqLen, InSize, Hidden, Wxh, Whh, Bh}`. Compile-time assertion `var _ layer.Layer[float64] = (*SimpleRNN[float64])(nil)`.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-15A02
- **Verify:**
  - `go build ./pkg/layer/recurrent/` clean.
  - `go test ./pkg/layer/recurrent -run TestSimpleRNNForward -v` — 3 fixed-seed forward passes match expected outputs within `1e-12` for `T=float64`.
  - `go test ./pkg/layer/recurrent -run TestSimpleRNNRoundTrip -v` — Marshal then Unmarshal recovers identical weights byte-for-byte.
- **Handoff:** Triggers T-15A04 (LSTM extends SimpleRNN's BPTT pattern to 4 gates); also enables T-15T01 finite-difference gradient validation.

### [T-15A04] `pkg/layer/recurrent/lstm.go` — `LSTM[T]`

- **Spec:** [l1-recurrent-layers.md](../specifications/l1-recurrent-layers.md) REC-3, REC-5, REC-6, REC-7, REC-8, REC-9; [l2-recurrent-impl.md](../specifications/l2-recurrent-impl.md) §5.3
- **Method:** Struct fields per §5.3 verbatim with fused gate matrices `[4·Hidden, ·]`. `NewLSTM[T](seqLen, inSize, hidden int)` constructs. `Init(rng)` — Xavier for `Wx`, Orthogonal for `Wh` (4 independent orthogonal blocks of size `Hidden×Hidden` per gate, stacked), zero for `B` with **forget-gate bias initialised to 1.0** (standard LSTM init trick to encourage long-term memory at start). `Forward(input)` — per step: compute `gates = sigmoid_or_tanh(Wx·x_t + Wh·h_{t-1} + B)` using fused matmul; split into `i, f, g, o`; update `c_t = f ⊙ c_{t-1} + i ⊙ g`; output `h_t = o ⊙ tanh(c_t)`. Cache `lastHidden`, `lastCell`, `lastGateActiv` for BPTT. `Backward(upstream)` — walk reverse time, accumulate `gradWx`/`gradWh`/`gradB` per REC-5; gate gradients via chain rule through sigmoid/tanh derivatives. `Step(x)` / `ResetState()` per §5.10. `MarshalJSON`/`UnmarshalJSON` serialise `{Type:"LSTM", SeqLen, InSize, Hidden, Wx, Wh, B}` with length-parity validation in Unmarshal. Compile-time assertion `var _ layer.Layer[float64] = (*LSTM[float64])(nil)`.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-15A02, T-15A03 (BPTT pattern from SimpleRNN reused)
- **Verify:**
  - `go build ./pkg/layer/recurrent/` clean.
  - `go test ./pkg/layer/recurrent -run TestLSTMForward -v` — 3 fixed-seed forward passes match expected outputs within `1e-12` for `T=float64`.
  - `go test ./pkg/layer/recurrent -run TestLSTMForgetBiasInit -v` — verifies `B[Hidden..2*Hidden]` (forget-gate bias slice) is initialised to 1.0.
- **Handoff:** Enables T-15T01 finite-difference gradient validation for both SimpleRNN and LSTM; closes Track A foundation scope.

### [T-15B01] `pkg/compute/gpu/` umbrella + sentinel additions

- **Spec:** [l2-backend-gpu.md](../specifications/l2-backend-gpu.md) §5.1, §5.2, §5.6
- **Method:** Create `pkg/compute/gpu/doc.go` — pure-Go package doc + constants `VendorOpenCL = "opencl"`, `VendorCUDA = "cuda"`. Create `pkg/compute/gpu/unavailable.go` — pure-Go shim type `unavailableBackend[T]` with all `Backend[T]` methods returning `ErrBackendUnavailable`; exposed via `gpu.New[T](vendor string) (compute.Backend[T], error)` constructor. Add to `pkg/utils/errors.go`: `ErrBackendUnavailable`, `ErrBackendTransfer`, `ErrBackendKernel` (all wrapping the new `Compute` category sentinel; if `Compute` sentinel doesn't yet exist, add it adjacent to existing `IO`/`Config`/`Integrity` categories).
- **Status:** Todo
- **Assignment:** Agent
- **Verify:**
  - `go build ./pkg/compute/gpu` clean (pure-Go default).
  - `go test ./pkg/compute/gpu` clean (no cgo tags).
  - `pkg/utils/errors.go` exports the 3 new sentinels; `go test ./pkg/utils` clean.
- **Handoff:** Triggers T-15B02 (OpenCL skeleton depends on the umbrella's sentinel definitions).

### [T-15B02] `pkg/compute/gpu/opencl/` skeleton

- **Spec:** [l2-backend-gpu.md](../specifications/l2-backend-gpu.md) §5.1, §5.3, §5.4
- **Method:** Files in `pkg/compute/gpu/opencl/` with `//go:build cgo && opencl` tag: `bindings.go` — cgo header `#cgo LDFLAGS: -lOpenCL` + `#include <CL/cl.h>`; minimal binding for `clGetPlatformIDs`, `clCreateContext`, `clCreateCommandQueue`, `clCreateBuffer`, `clEnqueueWriteBuffer`, `clEnqueueReadBuffer`. `opencl.go` — `Backend[T]` struct with `ctx C.cl_context`, `queue C.cl_command_queue`. `buffer.go` — `Buffer[T]` wrapping `C.cl_mem` + host-side `length int`. `Allocate(size int)` calls `clCreateBuffer`; `Free` calls `clReleaseMemObject`; `Write(buf, src)` calls `clEnqueueWriteBuffer`; `Read(buf, dst)` calls `clEnqueueReadBuffer` (blocking). Registration: `init()` calls `compute.Register("opencl", initialise[T])`. `initialise()` runs the discovery handshake; returns `ErrBackendUnavailable` if `clGetPlatformIDs` returns 0 devices.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-15B01
- **Verify:**
  - `go build -tags 'opencl' ./pkg/compute/gpu/opencl` clean on CI runner with OpenCL ICD installed.
  - `go test -tags 'opencl' ./pkg/compute/gpu/opencl -run TestBufferRoundTrip -v` — `Allocate(64) → Write(src) → Read(dst)` recovers `src` byte-for-byte. SKIP locally if `clGetPlatformIDs` returns 0.
- **Handoff:** Triggers T-15B03 (Forward kernel needs the buffer R/W infrastructure).

### [T-15B03] `pkg/compute/gpu/opencl/` Dense Forward kernel

- **Spec:** [l2-backend-gpu.md](../specifications/l2-backend-gpu.md) §4 COMP-1 + COMP-4, §5.3, §5.7
- **Method:** `pkg/compute/gpu/opencl/kernels.cl` — OpenCL C99 kernel `dense_forward(global T* weights, global T* bias, global T* input, global T* output, int inSize, int outSize)` — one work-item per output neuron, computes `sum + bias` then applies sigmoid activation. `pkg/compute/gpu/opencl/kernels.go` — host-side program loader via `//go:embed kernels.cl`, compiles via `clCreateProgramWithSource` + `clBuildProgram`, caches kernel handle. `Backend[T].Forward(layer, input)` — `clSetKernelArg` for weights/bias/input/output handles, `clEnqueueNDRangeKernel` with `global = layer.Size`. No Backward in this phase.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-15B02
- **Verify:**
  - `go build -tags 'opencl' ./pkg/compute/gpu/opencl` clean.
  - `go test -tags 'opencl' ./pkg/compute/gpu/opencl -run TestDenseForwardVsCPU -v` — for `(inSize=8, outSize=4)` and `(inSize=64, outSize=10)` cases, cross-reference vs `pkg/compute/cpu` Forward within `1e-5` (float32) / `1e-12` (float64). SKIP locally if no device.
- **Handoff:** Closes Track B foundation scope; Backward + CUDA + fallback wiring deferred to Phase 16.

### [T-15C01] `pkg/aimeta/` grammar package

- **Spec:** [l2-aimeta-linter.md](../specifications/l2-aimeta-linter.md) §5.1, §5.2, §5.6
- **Method:** Files in `pkg/aimeta/`: `vocab.go` — `var AllowedFields = []string{"Purpose","Usage","Lifecycle","Concurrency","Errors","Related","Constraints","Implementations","Stability"}`; `var StabilityEnum = []string{"Stable","Experimental","Deprecated","Internal"}`; `var ConcurrencyEnum = []string{"Safe","ReadSafe","SingleGoroutine","NotSafe"}`; `var TierRequiredFields = map[Tier][]string{...}` per l2-ai-doc-metadata §4.1 matrix. `violation.go` — `Violation{Pos token.Position; Symbol, Rule, Message string}`; rule codes constants. `ast.go` — `ExtractBlock(*ast.CommentGroup) (Block, bool)` — locates `AI-Meta:` label, slices subsequent lines. `grammar.go` — `ParseBlock(text string) (Block, []Violation)` returns rule codes LABEL/INDENT/CAP/LAST/VOCAB/MULTI/ENUM/ARTIFACT (TIER is computed by caller with tier context). `lint.go` — `Check(pkgPath string, opts Options) ([]Violation, error)` orchestrates parser.ParseFile + extraction + ParseBlock + tier check + artifact byte scan. `testdata/` golden fixtures: `good_public_type.go`, `bad_vocab.go` (extra `Author:` field), `bad_cap.go` (>12 lines), `bad_multi.go` (multi-line value), `bad_artifact.go` (contains `INV-`).
- **Status:** Todo
- **Assignment:** Agent
- **Verify:**
  - `go build ./pkg/aimeta` clean.
  - `go test ./pkg/aimeta -run TestGoldenFiles -v` — each fixture file produces expected `[]Violation` count + rule codes.
  - `pkg/aimeta/` coverage ≥80%.
- **Handoff:** Triggers T-15C02 (CLI wraps `aimeta.Check`).

### [T-15C02] `cmd/lint-aimeta/` CLI binary

- **Spec:** [l2-aimeta-linter.md](../specifications/l2-aimeta-linter.md) §5.3, §5.5
- **Method:** Files in `cmd/lint-aimeta/`: `main.go` — flag parsing (`--json`, `--include-tests`, `--no-internal`, `--resolve` accepted but ignored in this phase with `// TODO Phase 16: wire resolver`), dispatch to `aimeta.Check`. `output.go` — text formatter (one violation per line: `file:line:col [RULE] message`); JSON formatter (newline-delimited Violation records via `encoding/json`). `exitcode.go` — `Exit(violations []Violation, err error)` returns 0/1/2/3 per §5.5. `main_test.go` — end-to-end smoke test: run binary against `testdata/clean_pkg/` (exit 0), against `testdata/violating_pkg/` (exit 1, expected violation count). NO `--resolve` flag wiring in this phase.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-15C01
- **Verify:**
  - `go build ./cmd/lint-aimeta` clean.
  - `./bin/lint-aimeta ./pkg/utils` exits 0 (after T-15C03 adds compliant AI-Meta blocks to `pkg/utils`).
  - `./bin/lint-aimeta --json ./testdata/violating_pkg` exits 1 with newline-delimited JSON on stdout.
- **Handoff:** Triggers T-15C03 (first per-package hook validates the full integration).

### [T-15C03] `pkg/utils/` AI-Meta annotations + first compliance hook

- **Spec:** [l2-aimeta-linter.md](../specifications/l2-aimeta-linter.md) §5.4; [l2-ai-doc-metadata.md](../specifications/l2-ai-doc-metadata.md) §8 (Phase 2 rollout: `pkg/utils`)
- **Method:** Add AI-Meta trailing block to exported symbols in `pkg/utils/`: type `Float`, sentinel errors (`ErrUserConfig`/`ErrIntegrity`/`ErrIO`/etc.), helper functions (`Xavier`/`HeNormal`/`Orthogonal`/`Logger`). Each block follows the §5 examples in l2-ai-doc-metadata.md exactly. Create `pkg/utils/aimeta_test.go` — `TestAIMetaCompliance` per §5.4 pattern, invokes `aimeta.Check(".", aimeta.Options{IncludeInternal: true})`, fails the test on any violation.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-15C01 (uses `aimeta.Check`); T-15C02 (CLI exit code semantics inform the test failure path)
- **Verify:**
  - `go test ./pkg/utils -run TestAIMetaCompliance -v` passes with zero violations.
  - `go doc ./pkg/utils` shows the AI-Meta block rendered as a clean labeled list.
- **Handoff:** Closes Track C foundation scope; rollout to remaining packages (`pkg/activation`, `pkg/loss`, etc. per l2-ai-doc-metadata §8) deferred to Phase 16+.

### [T-15T01] Recurrent gradient validation

- **Goal:** Verify BPTT correctness for SimpleRNN and LSTM against finite-difference gradient; verify `utils.Orthogonal` produces truly orthonormal matrices.
- **Method:** `pkg/layer/recurrent/simple_rnn_test.go::TestSimpleRNNGradient` — 3 cases `(SeqLen=2, InSize=3, Hidden=4)`, `(SeqLen=5, InSize=8, Hidden=4)`, `(SeqLen=10, InSize=1, Hidden=8)`. For each: random input + random upstream gradient, compute analytical gradient via `Backward`, compare to numerical gradient `(L(W+ε) - L(W-ε)) / (2ε)` with `ε=1e-5` for `T=float64`; pass if relative error `< 1e-4`. `pkg/layer/recurrent/lstm_test.go::TestLSTMGradient` — same pattern with 3 cases. `pkg/utils/init_test.go::TestOrthogonal` — for `n ∈ {4, 32, 128}`, assert Frobenius distance of `Q · Q^T` from identity `< 1e-10` (float64).
- **Status:** Todo

### [T-15T02] AI-Meta golden-file matrix

- **Goal:** Verify `pkg/aimeta` correctly identifies all 5 §6 anti-patterns + accepts compliant examples.
- **Method:** `pkg/aimeta/lint_test.go::TestGoldenFiles` — table-driven over `testdata/` fixtures: each `.go` file has a companion `.want.json` describing the expected `[]Violation` (file:line:col + Rule + Symbol). For each pair, `aimeta.Check` output is compared to the JSON. Fixtures cover: clean public type (0 violations); §6.1 SDD-artifact leakage (`ARTIFACT`); §6.2 vocab creep (`VOCAB`); §6.3 multi-line value (`MULTI`); §6.4 block in the middle (`LAST`); §6.5 line cap overshoot (`CAP`).
- **Status:** Todo

### [T-15T03] GPU shim fallback + Warn-log path

- **Goal:** Verify pure-Go `gpu.New[T]("opencl")` returns `ErrBackendUnavailable` and that `pkg/nn/compile.go` falls back to `cpu` with a `Warn` log.
- **Method:** `pkg/compute/gpu/unavailable_test.go::TestNewWithoutCGO` (no build tag) — assert `gpu.New[float64]("opencl")` returns nil backend + `ErrBackendUnavailable`. `pkg/nn/compile_test.go::TestBackendFallback` — construct `NN` with `WithBackend("opencl")` on a default build, install `slogtest` capture, assert resolved backend is `cpu` and that the captured log contains level=Warn, `requested=opencl`.
- **Status:** Todo

### [T-15Z01] Phase 15 gate

- **Goal:** Confirm v0.13.0 RC criteria: build, test, coverage, all three tracks landed cleanly.
- **Method:**
  - `go build ./...` clean (default tags, no cgo).
  - `go build -tags 'opencl' ./...` clean on the OpenCL CI runner (SKIP on default).
  - `go test ./...` all packages green (count packages — expect 22+ after Phase 15 additions: existing 19 + `pkg/aimeta` + `pkg/compute/gpu` + `pkg/compute/gpu/opencl` + `pkg/layer/recurrent` + `cmd/lint-aimeta`).
  - Coverage: `pkg/layer/recurrent/` ≥80%, `pkg/aimeta/` ≥80%, `pkg/compute/gpu/` ≥80% (excluding cgo-tagged files), `pkg/utils/` maintained ≥80% (delta from Orthogonal addition + AI-Meta annotations).
  - `CHANGELOG.md` v0.13.0 entry written documenting all three tracks.
  - `cmd/lint-aimeta` smoke run on `pkg/utils/` exits 0.
  - Deferred items recorded explicitly in Phase 15 outcome line: GRU, LastStep, `WithGradClipNorm`, `optimizer.ClipByGlobalNorm`, `pkg/nn` recurrent options + compile wiring, OpenCL Backward, CUDA path, GPU perf benchmark gate, `cmd/lint-aimeta` `--resolve` flag, `TestAIMetaCompliance` rollout phases 3-5.
- **Status:** Todo
