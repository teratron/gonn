---
phase: 1
name: "Foundation Rewrite (Track A)"
status: Todo
subsystem: "pkg/utils, pkg/neuron, pkg/layer, pkg/network"
requires: []
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
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

- [ ] [T-1A01] Define error sentinels and category constants in `pkg/utils/errors.go`
- [ ] [T-1A02] Implement error helper constructors (`Newf`, `Wrap`, location-hint formatter)
- [ ] [T-1A03] [Validation] Unit tests for `errors.Is` routing + forbidden-phrase guard (per C32)
- [ ] [T-1A04] Implement RNG plumbing in `pkg/utils/init.go` (`math/rand/v2.PCG`, seed contract)
- [ ] [T-1A05] Implement Xavier / He / Uniform sampling helpers
- [ ] [T-1A06] [Validation] Distribution + reproducibility tests; benchmark Sample hot path

### Track B — Neuron (after Track A)

- [ ] [T-1B01] Rewrite `pkg/neuron/cell/{core,input,bias,dense,output}.go` with generic `[T utils.Float]`
- [ ] [T-1B02] Add missing `pkg/neuron/cell/hidden.go` (`Hidden[T]` type)
- [ ] [T-1B03] Add C26 compile-time interface assertions for every concrete cell type
- [ ] [T-1B04] Rewrite `pkg/neuron/axon/axon.go` — restore `OutgoingCell`, swap legacy RNG for `math/rand/v2.PCG`
- [ ] [T-1B05] [Validation] Cell + axon contract tests (table-driven, ≥80% coverage)

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
- **Status:** Todo
- **Assignment:** Agent
- **Handoff:** T-1A02 consumes the sentinels via `Newf`/`Wrap`.
- **Notes:** Per C32, every returned error must wrap one of `ErrUserConfig`, `ErrIntegrity`, `ErrCompute`, `ErrControl`, `ErrIO`. Stdlib only (C29).

### [T-1A02] Implement error helper constructors

- **Spec:** [l2-errors-impl.md](../specifications/l2-errors-impl.md) §Constructors
- **Status:** Todo
- **Assignment:** Agent
- **Handoff:** Used by all subsequent packages for error wrapping.
- **Notes:** First sentence of every doc-comment starts with the identifier (C31). Forbidden phrases per C32 must be unit-tested (T-1A03).

### [T-1A03] [Validation] Errors-impl tests

- **Goal:** Verify `errors.Is` routing through every sentinel and reject forbidden phrases.
- **Method:** `go test -race ./pkg/utils/...` with table-driven cases.
- **Status:** Todo

### [T-1A04] RNG plumbing in `pkg/utils/init.go`

- **Spec:** [l2-init-impl.md](../specifications/l2-init-impl.md) §RNG
- **Status:** Todo
- **Assignment:** Agent
- **Handoff:** Consumed by T-1A05 (sampling) and T-1B04 (axon).
- **Notes:** Use `math/rand/v2.PCG`; honor seed contract from `l1-weight-initialization`. No `math/rand` (legacy).

### [T-1A05] Xavier / He / Uniform sampling helpers

- **Spec:** [l2-init-impl.md](../specifications/l2-init-impl.md) §Sampling
- **Status:** Todo
- **Assignment:** Agent
- **Handoff:** Consumed by layer constructors (T-1C02).

### [T-1A06] [Validation] Init-impl tests

- **Goal:** Reproducibility (same seed → same sequence); distribution mean/variance within tolerance; benchmark `Sample` hot path.
- **Method:** `go test -race -bench=. ./pkg/utils/...`
- **Status:** Todo

### [T-1B01] Cell core/input/bias/dense/output rewrite

- **Spec:** [l2-neuron-model.md](../specifications/l2-neuron-model.md) §Cell types
- **Status:** Todo
- **Assignment:** Agent
- **Handoff:** Required by T-1B02 (Hidden), T-1B03 (assertions), Track C (layer).
- **Notes:** Generic over `[T utils.Float]` (C25). No infinite recursion in `CalculateValue` — that defect is closed in T-1D04 at the network layer.

### [T-1B02] Add `Hidden[T]` cell type

- **Spec:** [l2-neuron-model.md](../specifications/l2-neuron-model.md) §Cell types
- **Status:** Todo
- **Assignment:** Agent
- **Notes:** Closes the missing-type half of blocker C-001.

### [T-1B03] C26 compile-time interface assertions

- **Spec:** [l2-neuron-model.md](../specifications/l2-neuron-model.md) §Interface contract
- **Status:** Todo
- **Assignment:** Agent
- **Notes:** Per C26: `var _ Cell[float32] = (*Dense[float32])(nil)` for every concrete type.

### [T-1B04] Axon rewrite — restore `OutgoingCell`, modernize RNG

- **Spec:** [l2-neuron-model.md](../specifications/l2-neuron-model.md) §Axon
- **Status:** Todo
- **Assignment:** Agent
- **Notes:** Closes the commented-out-OutgoingCell half of blocker C-001. Pulls RNG from T-1A04.

### [T-1B05] [Validation] Cell + axon tests

- **Goal:** Table-driven contract tests for every cell type and axon path; ≥80% coverage.
- **Method:** `go test -race -cover ./pkg/neuron/...`
- **Status:** Todo

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
