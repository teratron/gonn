---
phase: 8
name: LR Scheduling Extension + CLI Binary
status: Todo
subsystem: pkg/optimizer, cmd/gonn
requires:
  - phase-7 (Scheduler[T] interface + BindScheduler + LearningRateSetter[T] + 4 scheduler types complete)
provides: []
key_files:
  created: []
  modified: []
patterns_established: []
duration_minutes: ~
---

# Phase 8 — LR Scheduling Extension + CLI Binary

**Status:** Done
**Decomposed:** 2026-05-10
**Tasks:** 6 feature + 2 validation + 1 gate = 9 total
**Specs:** l2-lr-scheduling-impl v1.0.0, l2-cli-client v0.2.0
**Track order:** A (LR extension) and B (CLI binary) run in parallel;
                 T-8T01 after Track A; T-8T02 after Track B; Gate T-8Z01 after all tracks.

## Track A — LR Scheduling Extension (pkg/optimizer)

*Goal: Add ExponentialLR — the simplest deferred scheduler type. Closes the most common
single-param decay pattern missing from Phase 7. ReduceOnPlateau and OneCycleLR deferred
(require MetricScheduler interface design decision — see Backlog note).*
*Source: [l2-lr-scheduling-impl.md](../specifications/l2-lr-scheduling-impl.md) §5.3*

- [x] **T-8A01** — Implement `pkg/optimizer/exponential_lr.go`: `ExponentialLR[T]` —
  rate decays by factor `gamma` every step. Formula: lr₀ × gamma^t. LRS-1..LRS-6
  compliance (same pattern as `StepLR`). Default `Granularity()` = `PerEpoch`.
  Tests: single step, accumulation, gamma=1.0 no-op, gamma near-zero floor,
  `SaveState`/`LoadState` round-trip, `Reset()` restores lr₀.

## Track B — CLI Binary (cmd/gonn)

*Goal: Ship standalone `gonn` binary — train/query/verify/version subcommands, CSV input,
exit-code contract, `--json` machine-readable output, streaming for large datasets.*
*Source: [l2-cli-client.md](../specifications/l2-cli-client.md)*
*Parallel with Track A — no shared files.*

- [x] **T-8B01** — Create `cmd/gonn/` directory structure and `cmd/gonn/main.go`:
  `switch os.Args[1]` routing to `trainCmd()`, `queryCmd()`, `verifyCmd()`, `versionCmd()`.
  Each subcommand owns its own `flag.FlagSet`. Print usage on unknown subcommand → exit 2.
  Wire `--precision` flag: `float32` (default) vs `float64` generic dispatch.

- [x] **T-8B02** — Implement `trainCmd()` in `cmd/gonn/train.go`:
  flags `--config`, `--data`, `--out` (default `weights.json`), `--epochs`, `--resume`, `--json`.
  Flow: load config JSON → load dataset CSV → optionally restore checkpoint → call `nn.Train()` →
  write output weights → print final loss. Exit codes 2 (`ErrUserConfig`), 3 (`ErrInputData`),
  4 (`ErrTrainingFailure`), 6 (`ErrIO`) per spec §5.3.

- [x] **T-8B03** — Implement `queryCmd()` in `cmd/gonn/query.go` and `verifyCmd()` in
  `cmd/gonn/verify.go`.
  `query`: flags `--config`, `--weights`, `--input` (comma-separated), `--json`. Calls `nn.Query()`.
  `verify`: flags `--config`, `--weights`, `--data`, `--json`. Calls forward pass + loss only.
  Both share the same config+weights load helper (extract to `cmd/gonn/load.go`).

- [x] **T-8B04** — Implement `versionCmd()` in `cmd/gonn/version.go`: prints
  `gonn <module_version> (go<runtime_version>)`. `--json` → `{"version":"...","go_version":"..."}`.
  Implement exit-code mapping helper in `cmd/gonn/exitcode.go`: `errors.Is` against
  `pkg/utils` sentinels → `os.Exit(code)` per spec §5.3 table.

- [x] **T-8B05** — Dataset streaming integration in `cmd/gonn/csv.go`:
  `loadDataset(path string) (Dataset, error)` — if file stat size > 64 MB, use
  `pkg/dataset.NewCSVReader` (streaming); otherwise `pkg/dataset.LoadCSV` (full load).
  Column mapping: first `config.Inputs` columns = input vector, remaining = target vector.
  Row errors: `ErrInputData` with row number in message (spec §5.2).

## Validation Tasks

- [x] **T-8T01** — LR Scheduling validation (after Track A):
  - `go test -race ./pkg/optimizer/...` — all scheduler tests green.
  - `ExponentialLR` coverage ≥80%; overall `pkg/optimizer/` coverage maintained ≥80%.
  - Verify `SaveState`/`LoadState` round-trip produces bit-identical outputs.
  - Update `l2-lr-scheduling-impl.md` §5.3 deferred list: move `ExponentialLR` to implemented.
    Bump spec version to 1.1.0. Update INDEX.md entry accordingly.

- [x] **T-8T02** — CLI integration tests (after Track B):
  - `go build ./cmd/gonn/` — binary builds clean.
  - `go test -race ./cmd/gonn/...` — subcommand routing, flag parsing, exit codes tested.
  - End-to-end smoke: XOR train → query → verify pipeline using `examples/xor/` config.
  - `--json` output validated against spec §5.4 schemas.
  - Large file path: mock file > 64 MB threshold triggers streaming code path.
  - Coverage `cmd/gonn/`: ≥80%.

## Gate

- [x] **T-8Z01** — Phase 8 gate:
  - `go build ./...` clean.
  - `go test -race ./...` all green (no regressions from prior phases).
  - Every new package (`cmd/gonn/`, extended `pkg/optimizer/`) ≥80% line coverage.
  - `l2-lr-scheduling-impl.md` updated to v1.1.0 with ExponentialLR in implemented list.
  - `CHANGELOG.md` v0.7.0 entry written (ExponentialLR scheduler + gonn CLI binary).
  - `v0.7.0` git tag created per `l1-release-policy.md §5.4`.
