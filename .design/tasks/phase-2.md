---
phase: 2
name: "Public Facade Restoration (Track B)"
status: Active
subsystem: "pkg/nn"
requires:
  - "phase-1: pkg/utils, pkg/neuron, pkg/layer, pkg/network"
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
bootstrap: true
---

# Phase 2 Tasks — Public Facade Restoration (Track B)

**Phase:** 2
**Status:** Active `[Bootstrap]`
**Strategic Goal:** Build the public `pkg/nn.NN[T]` facade per `l2-nn-facade` v2.0.0 dual-style API (Builder + Functional Options). Restore `Train()` / `Query()` / `Verify()` as first-class operations on top of the Phase-1 `network.Network[T]`. Land the training loop (`l2-training-loop`) and the lifecycle state-machine (`l2-control-impl`).

## Bootstrap Notice

All three source specifications are RFC/Draft. Tasks below are tagged `[Bootstrap]` per the project convention adopted in Phase 1: implementation drives spec validation; promotion to `Stable` is performed at the Phase Gate after the code matches the spec end-to-end.

## Track Layout

| Track | Scope | Depends on | Parallel-with |
| :--- | :--- | :--- | :--- |
| A — Builder API | `pkg/nn/{config,builder}.go` | Phase 1 | — |
| B — Options API | `pkg/nn/{options,presets}.go` | Track A (shares Config[T]) | — |
| C — Train / Query / Verify | `pkg/nn/{train,query,verify}.go` | Track A | Track B |
| D — Lifecycle Control | `pkg/nn/control.go` | Track A | Tracks B, C |

## Atomic Checklist

### Track A — Builder API + Internal Config

- [ ] [T-2A01] [Bootstrap] Define `Config[T]` and `HiddenLayerSpec[T]` in `pkg/nn/config.go` (per `l2-nn-facade` §5.5)
- [ ] [T-2A02] [Bootstrap] Define `WeightInitMethod` constants (Xavier / He / Random) in `pkg/nn/config.go` (§5.6)
- [ ] [T-2A03] [Bootstrap] Implement `NewBuilder[T]()` entry point and the Uninitialized → Configuring state guard in `pkg/nn/builder.go`
- [ ] [T-2A04] [Bootstrap] Implement topology methods `Input` / `Dense` / `Hidden` (alias) / `Output` (`Output` no longer accepts `loss.Type` — breaking change vs v1)
- [ ] [T-2A05] [Bootstrap] Implement configuration methods `WithLearningRate` / `WithLoss` / `WithBias` / `WithWeightInit` / `WithLossLimit` / `WithMaxIterations`
- [ ] [T-2A06] [Bootstrap] Implement callback methods `WithEpochCallback` / `WithBatchCallback`
- [ ] [T-2A07] [Bootstrap] Implement `Compile()` validation rules (§5.7 hard errors + soft warnings); wrap each error via `utils.Newf(utils.ErrUserConfig, ...)`
- [ ] [T-2A08] [Bootstrap] Implement `MustCompile()` panic-wrapper
- [ ] [T-2A09] [Bootstrap] Implement post-Compile no-op + Logger.Warn behaviour for any builder/option mutation (preserves L1 INV-2)
- [ ] [T-2A10] [Bootstrap] [Validation] Builder API table-driven tests — state machine, validation rules, compile-error routing through ErrUserConfig

### Track B — Functional Options API + Presets

- [ ] [T-2B01] [Bootstrap] Define `Option[T]` type and `New[T](opts...) (*NN[T], error)` constructor in `pkg/nn/options.go`
- [ ] [T-2B02] [Bootstrap] Implement topology options `WithInput` / `WithHiddenLayer` / `WithOutput`
- [ ] [T-2B03] [Bootstrap] Implement configuration options mirroring Builder methods (`WithLearningRate`, `WithLoss`, `WithBias`, `WithWeightInit`, `WithLossLimit`, `WithMaxIterations`, callbacks)
- [ ] [T-2B04] [Bootstrap] Implement higher-order options `Sequential` / `DeepNetwork` / `StandardSetup`
- [ ] [T-2B05] [Bootstrap] Implement presets `PresetXOR` / `PresetMNIST` / `PresetRegression` in `pkg/nn/presets.go`
- [ ] [T-2B06] [Bootstrap] Implement `MustNew[T](opts...)` panic-wrapper
- [ ] [T-2B07] [Bootstrap] [Validation] Options API tests — parity with Builder (same Config[T] state, same Compile path), preset round-trip

### Track C — Train / Query / Verify

- [ ] [T-2C01] [Bootstrap] Implement `Query(input []T) ([]T, error)` forward-only inference in `pkg/nn/query.go`
- [ ] [T-2C02] [Bootstrap] Implement `Verify(input, target []T) (T, error)` (forward + loss, no weight update) in `pkg/nn/verify.go`
- [ ] [T-2C03] [Bootstrap] Implement training loop core in `pkg/nn/train.go` — single-step forward + backward + weight update via `network.Network[T].Train`
- [ ] [T-2C04] [Bootstrap] Implement multi-epoch loop with `MaxIterations` and `LossLimit` early-stopping criteria (per `l1-training-semantics`)
- [ ] [T-2C05] [Bootstrap] Implement min-loss snapshot + rollback (per `l2-training-loop` §snapshot mechanics)
- [ ] [T-2C06] [Bootstrap] Implement `EpochCallback` / `BatchCallback` invocation points (synchronous; long callbacks block training — documented)
- [ ] [T-2C07] [Bootstrap] [Validation] Train/Query/Verify tests — XOR convergence via the public facade, verify-without-update preserves weights, query is read-only

### Track D — Lifecycle Control

- [ ] [T-2D01] [Bootstrap] Define lifecycle state cell (atomic) in `pkg/nn/control.go` per `l2-control-impl`
- [ ] [T-2D02] [Bootstrap] Implement `Pause()` / `Resume()` / `Stop()` API on `*NN[T]`; route invalid transitions through `utils.Newf(utils.ErrControl, ...)`
- [ ] [T-2D03] [Bootstrap] Implement safe-point check inside Train() loop (between epochs); honour Pause / Stop atomically
- [ ] [T-2D04] [Bootstrap] [Validation] Concurrency tests — pause from one goroutine while another is training; race-detector clean

### Phase Gate

- [ ] [T-2Z01] [Validation] `go build ./...` passes — public facade compiles end-to-end
- [ ] [T-2Z02] [Validation] `go test -race -cover ./pkg/nn/...` passes; coverage ≥ 80% on `pkg/nn`
- [ ] [T-2Z03] Promote `l2-nn-facade` RFC v2.0.0 → Stable; promote `l2-training-loop` and `l2-control-impl` Draft → Stable v1.0.0 once their implementations validate the contracts
- [ ] [T-2Z04] Update STATE.md, write CHANGELOG Phase 2 entries, populate phase-2 frontmatter (provides / key_files / patterns_established)

## Detailed Tracking

### [T-2A01] Define Config[T] and HiddenLayerSpec[T]

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.5
- **Status:** Todo
- **Notes:** Internal-only struct in v2.0; promotes to public when `l1-network-persistence` reaches Stable. Stdlib only.

### [T-2A02] WeightInitMethod constants

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.6
- **Status:** Todo
- **Notes:** String-typed constants (`xavier` / `he` / `random`). Default = `WeightInitXavier` enforced in Compile().

### [T-2A03] NewBuilder[T] + state machine

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.1, §5.2
- **Status:** Todo
- **Notes:** Three-state machine Uninitialized → Configuring → Operational. Illegal calls are no-ops with `Logger.Warn`, not panics.

### [T-2A04] Topology methods (Input / Dense / Hidden / Output)

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.2
- **Status:** Todo
- **Notes:** Hidden is an alias for Dense. Output drops loss.Type (breaking vs v1.0.0).

### [T-2A05] Configuration methods (WithX)

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.2
- **Status:** Todo
- **Notes:** All accept any chain position before Compile.

### [T-2A06] Callback methods

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.2
- **Status:** Todo
- **Notes:** Synchronous; long callbacks block training (documented, not enforced).

### [T-2A07] Compile() validation rules

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.7
- **Status:** Todo
- **Notes:** Hard errors → ErrUserConfig wrapped via utils.Newf. Soft warnings via Logger.Warn. Defaults applied per pseudo-code in §5.2.

### [T-2A08] MustCompile()

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.2
- **Status:** Todo
- **Notes:** Panic-wrapper for examples and tests.

### [T-2A09] Post-Compile no-op + Logger.Warn

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.1, §5.2 (Constraints), §4 (INV-2)
- **Status:** Todo
- **Notes:** Preserves L1 INV-2 (immutable topology after Compile).

### [T-2A10] [Validation] Builder API tests

- **Goal:** State machine correctness; every §5.7 hard error reachable; ErrUserConfig routing.
- **Method:** `go test -race -cover ./pkg/nn/...`
- **Status:** Todo

### [T-2B01] Option[T] + New[T] constructor

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.3
- **Status:** Todo
- **Notes:** New[T] runs Compile() implicitly — returns Operational network.

### [T-2B02] Topology options

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.3
- **Status:** Todo

### [T-2B03] Configuration options

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.3
- **Status:** Todo

### [T-2B04] Higher-order options

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.3
- **Status:** Todo
- **Notes:** Pure compositions of base options. May land in 0.1.x → 0.2.x patch without re-promoting the spec.

### [T-2B05] Presets

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.3
- **Status:** Todo
- **Notes:** PresetXOR matches the Phase-1 smoke-test topology — useful regression anchor.

### [T-2B06] MustNew[T]

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.3
- **Status:** Todo

### [T-2B07] [Validation] Options API tests

- **Goal:** Builder and Options must produce identical Config[T] state for the canonical XOR network. Preset round-trip identity.
- **Method:** `go test -race -cover ./pkg/nn/...`
- **Status:** Todo

### [T-2C01] Query()

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §4 (INV-3), [l2-network-graph.md](../specifications/l2-network-graph.md) §5.2
- **Status:** Todo

### [T-2C02] Verify()

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §4 (INV-3 + loss)
- **Status:** Todo
- **Notes:** Forward + loss; no backward, no weight update.

### [T-2C03] Train() core

- **Spec:** [l2-training-loop.md](../specifications/l2-training-loop.md), [l2-network-graph.md](../specifications/l2-network-graph.md)
- **Status:** Todo
- **Notes:** Single-step delegate; multi-epoch loop in T-2C04.

### [T-2C04] Multi-epoch loop with early stopping

- **Spec:** [l2-training-loop.md](../specifications/l2-training-loop.md), [l1-training-semantics.md](../specifications/l1-training-semantics.md)
- **Status:** Todo

### [T-2C05] Min-loss snapshot + rollback

- **Spec:** [l2-training-loop.md](../specifications/l2-training-loop.md) §snapshot
- **Status:** Todo
- **Notes:** Snapshot weights when loss reaches a new minimum; rollback to the snapshot at training-end if final loss diverged.

### [T-2C06] Epoch + Batch callbacks

- **Spec:** [l2-nn-facade.md](../specifications/l2-nn-facade.md) §5.2
- **Status:** Todo

### [T-2C07] [Validation] Train/Query/Verify tests

- **Goal:** XOR converges via NN.Train; Query is read-only; Verify preserves weights.
- **Method:** `go test -race -cover ./pkg/nn/...`
- **Status:** Todo

### [T-2D01] Lifecycle state cell

- **Spec:** [l2-control-impl.md](../specifications/l2-control-impl.md)
- **Status:** Todo
- **Notes:** `sync/atomic` int32 representing Idle / Running / Paused / Stopped.

### [T-2D02] Pause / Resume / Stop API

- **Spec:** [l2-control-impl.md](../specifications/l2-control-impl.md)
- **Status:** Todo
- **Notes:** Invalid transitions wrap ErrControl per Phase-1 taxonomy.

### [T-2D03] Safe-point check inside Train()

- **Spec:** [l2-control-impl.md](../specifications/l2-control-impl.md)
- **Status:** Todo
- **Notes:** Check between epochs; honour Pause (block) and Stop (return) atomically. No partial-batch interruption.

### [T-2D04] [Validation] Concurrency tests

- **Goal:** Pause from goroutine A while goroutine B is training. Race-detector clean.
- **Method:** `go test -race ./pkg/nn/...`
- **Status:** Todo

### [T-2Z01] Compile gate

- **Goal:** `go build ./...` exits 0 after the new facade lands.
- **Method:** `go build ./...`
- **Status:** Todo

### [T-2Z02] Test + coverage gate

- **Goal:** Race-detector clean; `pkg/nn` coverage ≥ 80% (C30).
- **Method:** `go test -race -cover ./pkg/nn/...`
- **Status:** Todo

### [T-2Z03] Spec promotion

- **Goal:** Promote l2-nn-facade RFC → Stable; l2-training-loop + l2-control-impl Draft → Stable v1.0.0 once the implementation matches the contract.
- **Method:** Manual `magic.spec` pass — bump headers and INDEX entries; add Document History rows.
- **Status:** Todo

### [T-2Z04] Phase wrap-up

- **Goal:** Update STATE.md, append CHANGELOG Phase 2 entries, populate phase-2 frontmatter.
- **Status:** Todo
