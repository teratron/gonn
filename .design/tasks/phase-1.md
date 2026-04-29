---
phase: 1
name: "Foundation Rewrite (Track A)"
status: In Progress
subsystem: "pkg/utils, pkg/neuron, pkg/layer, pkg/network"
requires: []
provides:
  - "pkg/utils — 6-sentinel error taxonomy + Newf/Wrap helpers"
  - "pkg/utils — math/rand/v2.PCG RNG factory + Xavier/He/Uniform samplers"
key_files:
  created:
    - pkg/utils/errors.go
    - pkg/utils/errors_test.go
    - pkg/utils/init.go
    - pkg/utils/init_test.go
    - pkg/neuron/cell/hidden.go
    - pkg/neuron/cell/cell_test.go
    - pkg/neuron/axon/axon_test.go
  modified:
    - pkg/neuron/cell/core.go
    - pkg/neuron/cell/input.go
    - pkg/neuron/cell/bias.go
    - pkg/neuron/cell/dense.go
    - pkg/neuron/cell/output.go
    - pkg/neuron/axon/axon.go
    - .design/specifications/l2-errors-impl.md
    - .design/INDEX.md
    - .design/PLAN.md
patterns_established:
  - "Hybrid 6-category error taxonomy (orthogonal): ErrUserConfig / ErrInputData / ErrCompute / ErrControl / ErrIntegrity / ErrIO"
  - "Multi-%w fmt.Errorf wrapping in Wrap() preserves both category and cause for errors.Is routing"
  - "Generic samplers [T utils.Float] computing in float64 then converting — zero-alloc hot path"
  - "Bootstrap-tagged tasks: RFC L2 specs are working contract; promotion to Stable deferred to phase gate"
  - "Generic type alias for type identity: type Hidden[T utils.Float] = Dense[T] (Go 1.24+) — full method inheritance without duplication"
  - "Dual axon constructors: New (default U[-0.5, 0.5] via package PCG + mutex) for legacy callers; NewWithWeight (caller-supplied) for layer-driven Xavier/He"
  - "Recursion-safe method shadowing: o.Dense.CalculateValue() in Output bypasses promotion-based recursion"
duration_minutes: ~
---

# Phase 1 Tasks — Foundation Rewrite (Track A)

**Phase:** 1
**Status:** Todo
**Strategic Goal:** Resolve compilation blocker C-001. Rewrite `pkg/neuron`, `pkg/layer`, `pkg/network` under fresh L2 contracts. Introduce `pkg/utils/errors.go` (sentinel taxonomy) and `pkg/utils/init.go` (RNG + sampling). Achieve `go build ./...` and `go test -race ./...` green by phase end.

## Track Layout

| Track | Scope | Depends on | Parallel-with |
| :--- | :--- | :--- | :--- |
| A — Foundations | `pkg/utils/{errors,init}.go` | — (leaves) | each other |
| B — Neuron | `pkg/neuron/cell/*`, `pkg/neuron/axon/*` | Track A | — |
| C — Layer | `pkg/layer/*` | Track B | — |
| D — Network | `pkg/network/*` | Track C | — |

## Atomic Checklist

### Track A — Foundations (parallel leaves)

- [x] [T-1A01] [Bootstrap] Define error sentinels and category constants in `pkg/utils/errors.go`
- [x] [T-1A02] [Bootstrap] Implement error helper constructors (`Newf`, `Wrap`, location-hint formatter)
- [x] [T-1A03] [Bootstrap] [Validation] Unit tests for `errors.Is` routing + forbidden-phrase guard (per C32)
- [x] [T-1A04] [Bootstrap] Implement RNG plumbing in `pkg/utils/init.go` (`math/rand/v2.PCG`, seed contract)
- [x] [T-1A05] [Bootstrap] Implement Xavier / He / Uniform sampling helpers
- [x] [T-1A06] [Bootstrap] [Validation] Distribution + reproducibility tests; benchmark Sample hot path

### Track B — Neuron (after Track A)

- [x] [T-1B01] Rewrite `pkg/neuron/cell/{core,input,bias,dense,output}.go` with generic `[T utils.Float]`
- [x] [T-1B02] Add missing `pkg/neuron/cell/hidden.go` (`Hidden[T]` type)
- [x] [T-1B03] Add C26 compile-time interface assertions for every concrete cell type
- [x] [T-1B04] Rewrite `pkg/neuron/axon/axon.go` — restore `OutgoingCell`, swap legacy RNG for `math/rand/v2.PCG`
- [x] [T-1B05] [Validation] Cell + axon contract tests (table-driven, ≥80% coverage)

### Track C — Layer (after Track B)

- [ ] [T-1C01] Rewrite `pkg/layer/{core,base}.go` — fix nil-deref constructors, dedupe `Init`
- [ ] [T-1C02] Implement `pkg/layer/{input,dense,output}.go` aligned to `l2-layer-types` v1.1.0
- [ ] [T-1C03] [Validation] Layer contract + nil-safety tests

### Track D — Network (after Track C)

- [ ] [T-1D01] Rewrite `pkg/network/network.go` — element-wise iteration over `Network[T]`
- [ ] [T-1D02] Rewrite `pkg/network/bundle.go` and complete `Build()` per `l2-network-graph` v1.1.0
- [ ] [T-1D03] Rewrite `pkg/network/propagation.go` — forward + backward passes
- [ ] [T-1D04] Fix `Output.CalculateValue` infinite recursion (cycle break per spec §5.4)
- [ ] [T-1D05] [Validation] Network smoke test (XOR convergence) + race-detector pass

### Phase Gate

- [ ] [T-1Z01] [Validation] `go build ./...` passes — C-001 resolved
- [ ] [T-1Z02] [Validation] `go test -race ./...` passes; `go test -cover` ≥ 80% on new packages
- [ ] [T-1Z03] Update STATE.md: clear blocker C-001, record patterns_established in phase frontmatter

## Detailed Tracking

### [T-1A01] Define error sentinels and category constants

- **Spec:** [l2-errors-impl.md](../specifications/l2-errors-impl.md) §Sentinels
- **Status:** Done
- **Assignment:** Agent
- **Handoff:** T-1A02 consumes the sentinels via `Newf`/`Wrap`.
- **Notes:** Per C32, every returned error must wrap one of the 6 orthogonal sentinels from `l2-errors-impl` v0.3.0 §5.1: `ErrUserConfig`, `ErrInputData`, `ErrCompute`, `ErrControl`, `ErrIntegrity`, `ErrIO`. Stdlib only (C29).
- **Changes:** [Bootstrap] Created `pkg/utils/errors.go` with 6 orthogonal sentinels and full C31 doc-comments.

### [T-1A02] Implement error helper constructors

- **Spec:** [l2-errors-impl.md](../specifications/l2-errors-impl.md) §Constructors
- **Status:** Done
- **Assignment:** Agent
- **Handoff:** Used by all subsequent packages for error wrapping.
- **Notes:** First sentence of every doc-comment starts with the identifier (C31). Forbidden phrases per C32 must be unit-tested (T-1A03).
- **Changes:** [Bootstrap] Added `Newf`, `Wrap` (multi-`%w`), `NewSizeError`, `NewActivationError`, `NewIntegrityError`, `LocationHint`. Wrap returns nil on nil cause; nil category panics.

### [T-1A03] [Validation] Errors-impl tests

- **Goal:** Verify `errors.Is` routing through every sentinel and reject forbidden phrases.
- **Method:** `go test -race ./pkg/utils/...` with table-driven cases.
- **Status:** Done
- **Changes:** [Bootstrap] `pkg/utils/errors_test.go` — orthogonality matrix, Newf routing, Wrap nil-cause + cause-preservation, panic-on-nil-cat, all helpers, C32 forbidden-phrase sweep, LocationHint, internal helpers. 100% line coverage. `-race` deferred to T-1Z02 (no gcc in dev env).

### [T-1A04] RNG plumbing in `pkg/utils/init.go`

- **Spec:** [l2-init-impl.md](../specifications/l2-init-impl.md) §RNG
- **Status:** Done
- **Assignment:** Agent
- **Handoff:** Consumed by T-1A05 (sampling) and T-1B04 (axon).
- **Notes:** Use `math/rand/v2.PCG`; honor seed contract from `l1-weight-initialization`. No `math/rand` (legacy).
- **Changes:** [Bootstrap] `NewRNG(seed) (*rand.Rand, uint64)` — zero seed → wall-clock fallback returning effective seed; PCG streams differentiated by golden-ratio xor.

### [T-1A05] Xavier / He / Uniform sampling helpers

- **Spec:** [l2-init-impl.md](../specifications/l2-init-impl.md) §Sampling
- **Status:** Done
- **Assignment:** Agent
- **Handoff:** Consumed by layer constructors (T-1C02).
- **Changes:** [Bootstrap] `XavierUniform[T Float]`, `HeNormal[T Float]`, `Uniform[T Float]` — 0 alloc/op, ~15-42 ns/op on 11th-gen i5; degenerate-fan fallback to Uniform.

### [T-1A06] [Validation] Init-impl tests

- **Goal:** Reproducibility (same seed → same sequence); distribution mean/variance within tolerance; benchmark `Sample` hot path.
- **Method:** `go test -race -bench=. ./pkg/utils/...`
- **Status:** Done
- **Changes:** [Bootstrap] `pkg/utils/init_test.go` — reproducibility 256 steps, Xavier/He/Uniform mean+variance within 5% over n=20000, float32 + float64 paths, nil-rng panic, degenerate-fan fallback, 4 zero-alloc benchmarks via `b.Loop()`.

### [T-1B01] Cell core/input/bias/dense/output rewrite

- **Spec:** [l2-neuron-model.md](../specifications/l2-neuron-model.md) §Cell types
- **Status:** Done
- **Assignment:** Agent
- **Handoff:** Required by T-1B02 (Hidden), T-1B03 (assertions), Track C (layer).
- **Notes:** Generic over `[T utils.Float]` (C25). Output recursion closed here at the cell layer via explicit `o.Dense.CalculateValue()` (Track-D recursion-fix entry T-1D04 retained for the network-side guard).
- **Changes:** Rewrote core/input/bias/dense/output.go with full C31 doc-comments. Renamed `_NewBias` → `NewBias`. Output.CalculateValue now delegates to embedded Dense and computes residual `target - value` on non-nil target.

### [T-1B02] Add `Hidden[T]` cell type

- **Spec:** [l2-neuron-model.md](../specifications/l2-neuron-model.md) §Cell types
- **Status:** Done
- **Assignment:** Agent
- **Notes:** Closes the missing-type half of blocker C-001.
- **Changes:** Added `pkg/neuron/cell/hidden.go` with `type Hidden[T utils.Float] = Dense[T]` (Go 1.24+ generic alias) and `NewHidden` constructor.

### [T-1B03] C26 compile-time interface assertions

- **Spec:** [l2-neuron-model.md](../specifications/l2-neuron-model.md) §Interface contract
- **Status:** Done
- **Assignment:** Agent
- **Notes:** Per C26: `var _ Cell[float32] = (*Dense[float32])(nil)` for every concrete type.
- **Changes:** Each cell file owns its own `var _ neuron.{Nucleus,Neuron}[float32|float64] = (*Type[...])(nil)` block, including the new Hidden alias.

### [T-1B04] Axon rewrite — restore `OutgoingCell`, modernize RNG

- **Spec:** [l2-neuron-model.md](../specifications/l2-neuron-model.md) §Axon
- **Status:** Done
- **Assignment:** Agent
- **Notes:** Closes the commented-out-OutgoingCell half of blocker C-001. Pulls RNG from T-1A04.
- **Changes:** Restored `OutgoingCell neuron.Neuron[T]` field; swapped `math/rand` global for `math/rand/v2.PCG` via `utils.NewRNG`; introduced `NewWithWeight` for caller-supplied weights from layer constructors; package mutex serialises the default-init path.

### [T-1B05] [Validation] Cell + axon tests

- **Goal:** Table-driven contract tests for every cell type and axon path; ≥80% coverage.
- **Method:** `go test -race -cover ./pkg/neuron/...` (race deferred — no gcc).
- **Status:** Done
- **Changes:** `pkg/neuron/cell/cell_test.go` and `pkg/neuron/axon/axon_test.go`. 100 % line coverage on both packages. Includes regression test for Output recursion, concurrency probe over default-init RNG, float32 + float64 paths, baseline-range guard for default weights.

### [T-1C01] Layer base + core rewrite

- **Spec:** [l2-layer-types.md](../specifications/l2-layer-types.md) §Base
- **Status:** Todo
- **Assignment:** Agent
- **Notes:** Fixes nil-deref constructors and the `Init` dedup defect catalogued in spec §5.3.

### [T-1C02] Implement Input/Dense/Output layers

- **Spec:** [l2-layer-types.md](../specifications/l2-layer-types.md) §Concrete types
- **Status:** Todo
- **Assignment:** Agent
- **Notes:** Pulls sampling helpers from T-1A05.

### [T-1C03] [Validation] Layer contract tests

- **Goal:** Verify nil-safety, `Init` idempotence, and forward/backward shape contracts.
- **Method:** `go test -race -cover ./pkg/layer/...`
- **Status:** Todo

### [T-1D01] `Network[T]` element-wise iteration rewrite

- **Spec:** [l2-network-graph.md](../specifications/l2-network-graph.md) §Iteration
- **Status:** Todo
- **Assignment:** Agent

### [T-1D02] Bundle + `Build()` completion

- **Spec:** [l2-network-graph.md](../specifications/l2-network-graph.md) §Build
- **Status:** Todo
- **Assignment:** Agent
- **Notes:** Required handoff for `pkg/nn.Builder` in Phase 2.

### [T-1D03] Forward + backward propagation

- **Spec:** [l2-network-graph.md](../specifications/l2-network-graph.md) §Propagation
- **Status:** Todo
- **Assignment:** Agent

### [T-1D04] Fix `Output.CalculateValue` infinite recursion

- **Spec:** [l2-network-graph.md](../specifications/l2-network-graph.md) §5.4
- **Status:** Todo
- **Assignment:** Agent
- **Notes:** This is the third half of blocker C-001 — the recursion lives at the network/output boundary, not inside a single cell.

### [T-1D05] [Validation] Network smoke test

- **Goal:** XOR convergence inside `loss-limit` and stable mean-square error.
- **Method:** `go test -race -cover ./pkg/network/...`
- **Status:** Todo

### [T-1Z01] [Validation] Compile gate

- **Goal:** `go build ./...` exits 0 — closes C-001.
- **Method:** `go build ./...`
- **Status:** Todo

### [T-1Z02] [Validation] Test + coverage gate

- **Goal:** Race-detector clean; coverage ≥80% on new packages (C30).
- **Method:** `go test -race -cover ./...`
- **Status:** Todo

### [T-1Z03] Phase wrap-up — STATE.md and frontmatter

- **Goal:** Clear blocker C-001 from STATE.md; populate `provides`, `key_files`, `patterns_established`, `duration_minutes` in this file's frontmatter.
- **Status:** Todo
