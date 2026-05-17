---
phase: 14
name: "Conv2D Implementation + MNIST CNN Example"
status: Todo
subsystem: "pkg/layer/conv/ (new conv2d/pool2d/flatten2d); pkg/nn/ (options + compile); pkg/dataset/ (MNIST adapter); examples/mnist_cnn/"
requires:
  - "Phase 13 ✓ (l2-conv-2d-impl Stable v0.1.0)"
  - "l1-conv-2d-layers Stable v0.2.0"
  - "l2-conv-2d-impl Stable v0.1.0"
  - "sibling pkg/layer/conv/ Conv1D + Flatten Stable (Phase 11)"
  - "pkg/dataset/mnist.go MNISTLoader[T] (Phase 11 Track C)"
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 14 Tasks — Conv2D Implementation + MNIST CNN Example

**Phase:** 14
**Status:** Todo
**Strategic Goal:** Realise the L2 contract `l2-conv-2d-impl.md` Stable v0.1.0 as working Go code in `pkg/layer/conv/` (2-D primitives), wire the new options through `pkg/nn/`, add the MNIST 2-D adapter to `pkg/dataset/`, and ship `examples/mnist_cnn/` (E16) as the canonical CV demo. Target release: v0.12.0.

## Atomic Checklist

### Track A — Conv2D Primitives (parallel-safe: A01/A03/A04 independent; A02 follows A01)

- [ ] [T-14A01] `pkg/layer/conv/conv2d.go` — Conv2D[T] type + NewConv2D + Forward + Init + outputShape + Validate
- [ ] [T-14A02] `pkg/layer/conv/conv2d.go` — Conv2D[T].Backward + GradSlots + MarshalJSON/UnmarshalJSON
- [ ] [T-14A03] `pkg/layer/conv/pool2d.go` — MaxPool2D[T] + AvgPool2D[T] with argmax tracking
- [ ] [T-14A04] `pkg/layer/conv/flatten2d.go` — Flatten2D[T] stateless CHW collapse + reshape backward

### Track B — NN Integration (serial after Track A)

- [ ] [T-14B01] `pkg/nn/options.go` + `pkg/nn/config.go` — WithConv2D/WithMaxPool2D/WithAvgPool2D/WithFlatten2D + Conv2DPrefix field
- [ ] [T-14B02] `pkg/nn/compile.go` — prepend Conv2D prefix; chain outputShape() to size first Dense layer

### Track C — MNIST 2-D Adapter (parallel with Track A)

- [ ] [T-14C01] Amend `l2-dataset-loader-impl.md` (patch v0.1.0 → v0.1.1) — add WithImageShape requirement section
- [ ] [T-14C02] `pkg/dataset/mnist.go` — WithImageShape(channels, height, width) decorator + round-trip test

### Track D — MNIST CNN Example (serial after A+B+C)

- [ ] [T-14D01] `examples/mnist_cnn/main.go` + `examples/mnist_cnn/README.md` — E16 full CNN demo
- [ ] [T-14D02] Amend `l2-usage-examples.md` (minor v1.0.0 → v1.1.0) — register E16

### Validation + Gate

- [ ] [T-14T01] `pkg/layer/conv/conv2d_test.go` — full backward test matrix (CONV2D-4) with finite-difference gradient check
- [ ] [T-14Z01] Phase 14 gate — build + test + coverage + v0.12.0 release candidate

## Detailed Tracking

### [T-14A01] `pkg/layer/conv/conv2d.go` — Conv2D[T] type + NewConv2D + Forward + Init + outputShape + Validate

- **Spec:** [l2-conv-2d-impl.md](../specifications/l2-conv-2d-impl.md) §5.2 (Conv2D Weight Storage), §5.3 (outputShape helper); [l1-conv-2d-layers.md](../specifications/l1-conv-2d-layers.md) CONV2D-1, CONV2D-2, CONV2D-3, CONV2D-8, CONV2D-9
- **Method:** Generalise `pkg/layer/conv/conv1d.go` to 2-D. Struct fields per §5.2 verbatim. Forward implements CHW row-major (CONV2D-C9): for each filter `f`, for each output position `(oy, ox)`, accumulate `sum_{c, ki, kj} Weights[f,c,ki,kj] * x[c, oy*sH+ki-padTop, ox*sW+kj-padLeft]`. Init calls `utils.HeNormal[T](rng, InChannels*KernelH*KernelW)`. Add compile-time `var _ layer.Layer[float64] = (*Conv2D[float64])(nil)` assertion in `conv.go`.
- **Status:** Todo
- **Assignment:** Agent
- **Verify:**
  - `go build ./pkg/layer/conv/...` clean.
  - `go test ./pkg/layer/conv/ -run TestConv2DForward -v` — table-driven Forward test passes for `(C_in=1, F=2, K=3, S=1, PadValid)` and `(C_in=3, F=4, K=3, S=2, PadSame)`.
  - `outputShape(28, 28, 3, 3, 1, 1, PadValid)` returns `(26, 26)`; `outputShape(28, 28, 3, 3, 2, 2, PadSame)` returns `(14, 14)`.
- **Handoff:** Triggers T-14A02 (Backward path needs Forward semantics fixed).

### [T-14A02] `pkg/layer/conv/conv2d.go` — Backward + GradSlots + JSON

- **Spec:** [l2-conv-2d-impl.md](../specifications/l2-conv-2d-impl.md) §4 CONV2D-4 row; §5.2 grad buffer fields; [l1-conv-2d-layers.md](../specifications/l1-conv-2d-layers.md) CONV2D-4, CONV2D-7
- **Method:** 6-level loop nesting per CONV2D-4: `gradW[f,c,ki,kj] += lastInput[c, oy*sH+ki-padTop, ox*sW+kj-padLeft] * upstream[f,oy,ox]`; `gradX[c, ...] += Weights[f,c,ki,kj] * upstream[f,oy,ox]`; `gradB[f] += sum(upstream[f,:,:])`. Reuse cap-check + zero-reset pattern from `Conv1D.Backward` (PERF-4 zero-alloc steady-state). MarshalJSON via struct-alias pattern; UnmarshalJSON validates `len(Weights) == NumFilters*InChannels*KernelH*KernelW`.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-14A01
- **Verify:**
  - `go test ./pkg/layer/conv/ -run TestConv2DBackwardShape -v` — gradW/gradB/gradX have expected dimensions for `(C_in=2, F=3, K=3, H=W=5, PadValid)`.
  - JSON round-trip: marshal `Conv2D[float64]{NumFilters:4, InChannels:2, KernelH:3, KernelW:3, ...}`, unmarshal, verify `Weights`/`Biases`/shape fields bit-identical.
  - `go test -bench=BenchmarkConv2DBackward -benchmem` — 0 allocs/op after warm-up iterations (PERF-4 floor).
- **Handoff:** Triggers T-14T01 (numerical gradient check now meaningful).

### [T-14A03] `pkg/layer/conv/pool2d.go` — MaxPool2D[T] + AvgPool2D[T]

- **Spec:** [l2-conv-2d-impl.md](../specifications/l2-conv-2d-impl.md) §5.4 (Pool Backward); [l1-conv-2d-layers.md](../specifications/l1-conv-2d-layers.md) CONV2D-5
- **Method:** Non-overlapping windows (`PoolH × PoolW`), per-channel-independent. MaxPool2D stores `argmaxH []int` + `argmaxW []int` of length `C * outH * outW`; Backward routes `upstream[c,oy,ox]` only to `lastInput[c, argmaxH[idx], argmaxW[idx]]`. AvgPool2D Backward distributes `upstream[c,oy,ox] / (PoolH*PoolW)` to all positions in window. Both implement `layer.Layer[T]`. JSON: only `pool_h`/`pool_w` fields persist.
- **Status:** Todo
- **Assignment:** Agent
- **Verify:**
  - `go test ./pkg/layer/conv/ -run TestMaxPool2DBackward -v` — gradient routes only to argmax position (other positions get 0).
  - `go test ./pkg/layer/conv/ -run TestAvgPool2DBackward -v` — gradient distributed evenly (each position gets `upstream/(P*P)`).
  - Forward shape: `MaxPool2D(2,2)` on `(C=4, H=8, W=8)` returns `(C=4, H=4, W=4)`.
- **Handoff:** Independent — can run parallel with T-14A01/A02/A04.

### [T-14A04] `pkg/layer/conv/flatten2d.go` — Flatten2D[T]

- **Spec:** [l2-conv-2d-impl.md](../specifications/l2-conv-2d-impl.md) §5.5 (Flatten2D); [l1-conv-2d-layers.md](../specifications/l1-conv-2d-layers.md) CONV2D-6
- **Method:** Stateless beyond `lastC/lastH/lastW` shape memory. Forward: copy CHW-flat input to output (Flatten2D is the boundary between feature-map and Dense). Backward: reshape upstream gradient back to CHW shape using saved `lastC/lastH/lastW`. Element order MUST follow CONV2D-C9 flat-index formula `c*H*W + y*W + x`.
- **Status:** Todo
- **Assignment:** Agent
- **Verify:**
  - `go test ./pkg/layer/conv/ -run TestFlatten2DRoundTrip -v` — Forward then Backward returns same shape; element values bit-identical (Flatten is pure reshape).
  - CHW order verified: input `(C=2, H=3, W=4)` with element `(1, 2, 3)` set to 42 → flat index `1*12 + 2*4 + 3 = 23`; assert `output[23] == 42`.
- **Handoff:** Independent — can run parallel with T-14A01/A02/A03.

### [T-14B01] `pkg/nn/options.go` + `pkg/nn/config.go` — Options + Conv2DPrefix

- **Spec:** [l2-conv-2d-impl.md](../specifications/l2-conv-2d-impl.md) §5.6 (Options Wiring); [l2-nn-facade.md](../specifications/l2-nn-facade.md) §Functional Options
- **Method:** Add four `Option[T]` constructors: `WithConv2D[T](numFilters, inChannels, kernelH, kernelW, strideH, strideW int, pad PadMode, useBias bool)`, `WithMaxPool2D[T](poolH, poolW int)`, `WithAvgPool2D[T](poolH, poolW int)`, `WithFlatten2D[T]()`. Each appends to a new `Conv2DPrefix []layer.Layer[T]` field on `config`. Match Builder fluent API style: optional `Builder.WithConv2D(...)` chain methods for parity.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-14A01, T-14A03, T-14A04 (needs all four types defined)
- **Verify:**
  - `go build ./pkg/nn/...` clean.
  - `go test ./pkg/nn/ -run TestWithConv2D -v` — option appends correct layer type to `Conv2DPrefix`.
  - Builder + Functional Options parity test: both APIs produce identical `[]layer.Layer[T]` prefix from same input arguments.
- **Handoff:** Triggers T-14B02 (compile() must consume Conv2DPrefix).

### [T-14B02] `pkg/nn/compile.go` — Prepend Conv2D Prefix + Shape Chain

- **Spec:** [l2-conv-2d-impl.md](../specifications/l2-conv-2d-impl.md) §5.7 (compile() integration); [l1-conv-2d-layers.md](../specifications/l1-conv-2d-layers.md) CONV2D-1
- **Method:** In `compile()`, prepend `Conv2DPrefix` before existing `Conv1DPrefix` and Dense hidden stack. For each Conv2D layer, call `outputShape(inH, inW, kH, kW, sH, sW, pad)` to compute the next layer's spatial dimensions. After final `Flatten2D`, multiply `C_out * H_out * W_out` to set the first Dense layer's `InputSize`. Surface `ErrConv2DShapeMismatch` from `Conv2D.Validate` as a build-time error.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-14B01
- **Verify:**
  - `go test ./pkg/nn/ -run TestCompileConv2DChain -v` — `Input → Conv2D(8,1,3,3,1,1,PadValid) → MaxPool2D(2,2) → Conv2D(16,8,3,3,1,1,PadValid) → MaxPool2D(2,2) → Flatten2D → Dense(64) → Output(10)` on `(1, 28, 28)` input compiles cleanly; first Dense `InputSize` matches computed flat size.
  - Invalid shape (e.g., kernel larger than input under PadValid) returns wrapped `ErrConv2DShapeMismatch` at compile time, not runtime.
- **Handoff:** Triggers T-14C02 + T-14D01 (compile must accept the CNN architecture).

### [T-14C01] Amend `l2-dataset-loader-impl.md` — WithImageShape Requirement

- **Spec:** [l2-conv-2d-impl.md](../specifications/l2-conv-2d-impl.md) §6 Note 4 (MNIST 2-D adapter); [l2-dataset-loader-impl.md](../specifications/l2-dataset-loader-impl.md)
- **Method:** Patch bump (v0.1.0 → v0.1.1). Add §X "Image Shape Adapter" describing `WithImageShape(channels, height, width int)` decorator: wraps existing flat-byte tensor output without copying; produces CHW-layout view; one-time shape validation at decorator construction (`channels*height*width == flat_length`). Update Document History row with date + reason ("CNN integration for Phase 14"). Update INDEX.md version field.
- **Status:** Todo
- **Assignment:** Agent
- **Verify:**
  - `l2-dataset-loader-impl.md` `Version:` header reads `0.1.1`; new section present with `WithImageShape` signature + invariant ("flat byte tensor MUST round-trip with `c*H*W + y*W + x` ordering").
  - INDEX.md entry matches: `| [l2-dataset-loader-impl.md](specifications/l2-dataset-loader-impl.md) | ... | Stable | L2 | 0.1.1 |`.
  - `node .magic/scripts/executor.js check-prerequisites --json --verify-headers --workspace main` returns `ok: true` with no header drift.
- **Handoff:** Triggers T-14C02 (implementation follows spec amendment).

### [T-14C02] `pkg/dataset/mnist.go` — WithImageShape Decorator

- **Spec:** [l2-dataset-loader-impl.md](../specifications/l2-dataset-loader-impl.md) §Image Shape Adapter (post-T-14C01); [l1-dataset-formats.md](../specifications/l1-dataset-formats.md) §IDX format
- **Method:** Add `WithImageShape(channels, height, width int) DatasetOption[T]` (or equivalent). On `Next()`, the decorator validates `len(flat) == channels*height*width` and returns the same slice unchanged (no copy — CHW byte order matches MNIST IDX storage by default for single-channel `(1, 28, 28)`). For multi-channel datasets, document that callers must pre-transpose HWC → CHW upstream.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-14C01
- **Verify:**
  - `go test ./pkg/dataset/ -run TestMNISTWithImageShape -v` — adapter wraps `MNISTLoader[float64]` and produces `(1, 28, 28)` tensors; flat-byte round-trip preserves all 784 values.
  - `pkg/dataset/` coverage ≥80% (C30) after T-14C02.
- **Handoff:** Triggers T-14D01 (CNN example needs the adapter).

### [T-14D01] `examples/mnist_cnn/main.go` + `README.md` — E16 Demo

- **Spec:** [l2-usage-examples.md](../specifications/l2-usage-examples.md) §E16 (added in T-14D02); [l1-conv-2d-layers.md](../specifications/l1-conv-2d-layers.md) §4.2 Shape Propagation Example
- **Method:** Full CNN: `Input → Conv2D(8, 1, 3, 3, 1, 1, PadValid) → MaxPool2D(2, 2) → Conv2D(16, 8, 3, 3, 1, 1, PadValid) → MaxPool2D(2, 2) → Flatten2D → Dense(64) → Output(10)`. Use `WithImageShape(1, 28, 28)` on `MNISTLoader[float64]`. Train for 5 epochs with `SGD[float64]` + `WithBatchNorm`. README documents: download MNIST IDX files into `data/`, run `go run ./examples/mnist_cnn/`, expected accuracy ≥95% after 5 epochs. Smoke-run deferred to user-supplied data (same pattern as E06).
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-14A01..A04, T-14B01..B02, T-14C02
- **Verify:**
  - `go build ./examples/mnist_cnn/...` clean.
  - `go vet ./examples/mnist_cnn/` clean.
  - README.md links to `l2-usage-examples.md` §E16 + lists exact IDX filenames and download URL pattern (no committed binary data).
- **Handoff:** Triggers T-14D02 (catalog entry follows code shape).

### [T-14D02] Amend `l2-usage-examples.md` — Register E16

- **Spec:** [l2-usage-examples.md](../specifications/l2-usage-examples.md)
- **Method:** Minor bump (v1.0.0 → v1.1.0). Add `##### E16 — MNIST CNN (2-D convolutional)` entry after E15 with same template (Dataset / Topology / Hyperparams / Expected Output rows). Reference `examples/mnist_cnn/`. Update Document History + INDEX.md version.
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-14D01
- **Verify:**
  - `l2-usage-examples.md` `Version:` header reads `1.1.0`; new E16 row present with concrete shape `(1, 28, 28)` and topology spec.
  - INDEX.md matches: `| [l2-usage-examples.md](...) | ... | Stable | L2 | 1.1.0 |`.
- **Handoff:** Triggers T-14Z01 gate.

### [T-14T01] `pkg/layer/conv/conv2d_test.go` — Backward Test Matrix + Finite-Difference

- **Spec:** [l2-conv-2d-impl.md](../specifications/l2-conv-2d-impl.md) §6 Note 3 (Backward-pass test matrix); [l1-conv-2d-layers.md](../specifications/l1-conv-2d-layers.md) CONV2D-4
- **Method:** Table-driven test covering `{PadValid, PadSame} × {MaxPool2D, AvgPool2D} × {C_in=1, C_in≥2}` = 8 cases minimum. For each case, run forward → backward, then verify via centred finite-difference: `gradW_fd[i] = (f(W + ε*e_i) - f(W - ε*e_i)) / (2ε)` with `ε = 1e-4` (float32) / `ε = 1e-6` (float64); assert `|gradW_analytical - gradW_fd| < 1e-3` (float32) / `< 1e-9` (float64). At least one multi-channel case (`C_in=3`).
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** T-14A02, T-14A03
- **Verify:**
  - `go test ./pkg/layer/conv/ -run TestConv2DGradientFiniteDifference -v` — all 8 matrix cases pass.
  - `go test ./pkg/layer/conv/ -race` — no data races (single-goroutine layer, but covers any sync.Pool wrapper).
  - `pkg/layer/conv/` coverage ≥80% (C30) for the new Conv2D code paths.
- **Handoff:** Triggers T-14Z01 gate.

### [T-14Z01] Phase 14 Gate — Build + Test + Coverage + Release

- **Goal:** Close Phase 14 with v0.12.0 release candidate ready.
- **Method:**
  - `go build ./...` clean (all packages).
  - `go test ./...` all packages green; record per-package coverage.
  - `pkg/layer/conv/` coverage ≥80%; `pkg/dataset/` coverage ≥80% (C30 floor).
  - Re-run `node .magic/scripts/executor.js check-prerequisites --json --workspace main` — must return `ok: true` with zero new warnings (persistent RFC `l2-ai-doc-metadata` excluded).
  - Update phase-14.md frontmatter: `status: Done`, populate `provides`, `key_files.created` + `key_files.modified`, `patterns_established`, `duration_minutes`.
  - Append CHANGELOG.md v0.12.0 entry under existing `[Unreleased]` (release tag deferred to user).
- **Status:** Todo
- **Assignment:** Agent
- **Depends on:** All Track A/B/C/D tasks + T-14T01
- **Verify:** `go build` + `go test` clean; `check-prerequisites` clean; CHANGELOG entry written; phase-14.md frontmatter complete; STATE.md `Next Action` points at user release tag.
- **Handoff:** After gate green → suggest `git tag -a v0.12.0` commit (user runs manually per memory rule).

## Notes

- **CHW layout is non-negotiable** — CONV2D-C9 flat-index formulae must be copied verbatim from `l1-conv-2d-layers.md` §2; any independent re-derivation is a regression risk.
- **PERF-4 zero-alloc backward** — Conv2D.Backward must hit zero allocs/op after the first iteration (sync.Pool-eligible scratch buffers). Track via `go test -bench=BenchmarkConv2DBackward -benchmem`.
- **C29 stdlib-only** — no new external dependencies. All math uses `math/rand/v2` + `pkg/utils/` helpers.
- **C30 coverage floor** — `pkg/layer/conv/` ≥80%; `pkg/dataset/` ≥80%; `pkg/nn/` regression: must remain ≥80% after compile.go changes.
- **MNIST IDX data NOT committed** — same as E06 pattern. README documents download URLs but never checks in binary data.
- **Race detector on Windows** — CGO required; tests run without `-race` locally per Phase 11 precedent. T-14T01 verification line includes `-race` for CI environments.
- **@role:planner audit findings** (Phase 14 scoping):
  - Optimism bias: Conv2D.Backward isolated to T-14A02 alone (not bundled with Forward) because 6-level loop nesting is the dominant complexity source.
  - Hidden deps: Track A→B serial (B needs A's types); Track C parallel with A; Track D waits A+B+C.
  - Cascade risk: Mandatory finite-difference gradient check in T-14T01 mitigates silent CONV2D-4 regressions that would otherwise surface only in failed MNIST CNN convergence.
- Engine drift `.magic/.version` = 2.1.27 vs INDEX snapshot 2.1.25 remains acknowledged (n-branch). Not a blocker for Phase 14.
