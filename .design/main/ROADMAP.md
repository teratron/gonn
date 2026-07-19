# Implementation Roadmap

**Version:** 1.0.0
**Status:** Active
**Decided:** 2026-04-28

## Strategic Path: Hybrid

Locked after the 2026-04-28 code-state analysis of `pkg/` and `examples/`.

- **Keep** (Stable, not modified): `pkg/activation/*` (10 functions + dispatcher + tests), `pkg/loss/*` (18 functions + dispatcher + tests), `pkg/utils/float.go` (`Float = float32 | float64`), `pkg/utils/logger.go` (`log/slog` wrapper).
- **Rewrite** (broken core): `pkg/nn/`, `pkg/network/`, `pkg/layer/`, `pkg/neuron/cell/`, `pkg/neuron/axon/`.
- **Add new** (per L2 partner specs): `pkg/persistence/`, `pkg/checkpoint/`, `pkg/dataset/`, `pkg/compute/{backend,registry,cpu/}`, `pkg/utils/errors.go`, `pkg/utils/init.go`.

### Why Hybrid

- The 28 files in `activation/` and `loss/` are not structurally broken — they are described by Stable L2 specs (`l2-activation-functions` v1.0.0, `l2-loss-functions` v1.0.0) and have working tests.
- The topology + training core has 12 catalogued critical defects (see `l2-network-graph` § 5.4, `l2-layer-types` § 5.3, `l2-neuron-model` § 5.4 "Known Issues") — including commented-out `Builder`, infinite recursion in `Output.CalculateValue`, type assertions that always fail, missing `cell.Hidden[T]` type. Patching is more expensive than rewriting under the new RFC contracts.
- Rewriting the math layer would risk numerical-equivalence regressions for zero architectural gain and force major-bump demotion of two Stable L2 specs.

## Implementation Tracks (input to `magic.task`)

### Track A — Foundation Rewrite

| # | Path | Spec source | Status of source |
| :--- | :--- | :--- | :--- |
| A.1 | `pkg/utils/errors.go` (new) | `l2-errors-impl` | RFC v0.6.0 |
| A.2 | `pkg/utils/init.go` (new) | `l2-init-impl` | RFC v0.6.0 |
| A.3 | `pkg/neuron/cell/{core,input,bias,dense,output,hidden}.go` (rewrite + add `hidden.go`) | `l2-neuron-model` | Stable v1.1.0 |
| A.4 | `pkg/neuron/axon/axon.go` (rewrite — restore `OutgoingCell`, modernize RNG to `math/rand/v2.PCG`) | `l2-neuron-model` | Stable v1.1.0 |
| A.5 | `pkg/layer/{core,base,input,dense,output}.go` (rewrite — fix nil-deref constructors, dedupe `Init`) | `l2-layer-types` | Stable v1.1.0 |
| A.6 | `pkg/network/{network,bundle,propagation}.go` (rewrite — element-wise iteration, complete `Build()`) | `l2-network-graph` | Stable v1.1.0 |

### Track B — Public Facade Restoration

| # | Path | Spec source | Status of source |
| :--- | :--- | :--- | :--- |
| B.1 | `pkg/nn/nn.go` (rewrite — `*NN[T]` with state machine) | `l2-nn-facade` | RFC v2.0.0 |
| B.2 | `pkg/nn/builder.go` (uncomment + align to v2.0.0 signatures) | `l2-nn-facade` | RFC v2.0.0 |
| B.3 | `pkg/nn/options.go` (new — functional options) | `l2-nn-facade` § Functional Options | RFC v2.0.0 |
| B.4 | `pkg/nn/train.go` (rewrite — convergence loop + safe-points) | `l2-training-loop` + `l2-control-impl` | Draft / Draft |
| B.5 | `pkg/nn/control.go` (new — atomic state cell, methods) | `l2-control-impl` | Draft |

### Track C — New Capability Packages

| # | Path | Spec source | Status of source |
| :--- | :--- | :--- | :--- |
| C.1 | `pkg/persistence/` | `l2-persistence-impl` | Draft |
| C.2 | `pkg/checkpoint/` | `l2-checkpointing-impl` | Draft |
| C.3 | `pkg/dataset/` | `l2-streaming-impl` | Draft |
| C.4 | `pkg/compute/{backend.go,registry.go,cpu/}` | `l2-backend-cpu` | Draft |
| C.5 | Performance scaffolding (`sync.Pool`, preallocation, `WithProfiling`) | `l2-perf-impl` | Draft |

### Track D — Examples Catalog

| # | Path | Spec source | Status of source |
| :--- | :--- | :--- | :--- |
| D.1 | `examples/E01_xor_minimal/` … `examples/E15_meta_learning/` (15 entries) | `l2-usage-examples` | RFC v1.0.0 |

## Build Order (suggested)

```text
A.1, A.2  ─────►  A.3, A.4  ─────►  A.5  ─────►  A.6  ─────►  B.1..B.5
   (parallel)        (sequential after errors+init)             (after foundation green)
                                                                       │
                                                                       ▼
                                                           C.1, C.2, C.3, C.4, C.5
                                                                  (parallel)
                                                                       │
                                                                       ▼
                                                                       D.1
                                                              (smoke-test catalog)
```

Rationale for ordering:

- A.1 (errors) and A.2 (init) are leaves — every other track needs them.
- A.3..A.6 follow the type-graph dependency (cell → axon → layer → network).
- B.* is the public API; cannot start before foundation compiles.
- C.* is independent of B but depends on A — can run in parallel with B once A is green.
- D.1 closes the loop; smoke tests in D.1 validate every track end-to-end.

## Pre-flight Gates (before `magic.task` invocation)

- [x] All specs in `INDEX.md` v1.7.0 are at least Draft (no holes in coverage).
- [x] All RFC L2 partners have a Stable L1 parent (errors, init, neuron, layer, network).
- [x] Code-state analysis recorded; defect catalogue complete.
- [x] Strategic path locked (hybrid).

## Constraints That Apply Across Tracks

- `RULES.md §C29` — stdlib-only; third-party deps require justification.
- `RULES.md §C30` — minimum 80% test coverage on every new file; benchmarks for hot paths.
- `RULES.md §C31` — doc-comment verbosity tier (audience-tiered).
- `RULES.md §C32` — error informativeness; forbidden phrases; wrap with sentinels.
- `CLAUDE.md §1.1` — all code, identifiers, comments, technical docs in English.

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-04-28 | Initial — hybrid path locked after code-state analysis. Tracks A–D defined as input to magic.task. |
