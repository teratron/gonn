# CLI Client for GoNN

**Version:** 0.2.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-neural-network-architecture.md

## Overview

A standalone command-line client `gonn` that wraps the library's public API for non-Go users and
CI pipelines. Provides `train` / `query` / `verify` / `version` subcommands operating on
JSON-described networks and CSV datasets, without requiring users to compile their own Go binary.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent
- [l2-nn-facade.md](l2-nn-facade.md) — Library API the CLI delegates to
- [l1-network-persistence.md](l1-network-persistence.md) — JSON schema the CLI reads/writes
- [l1-checkpointing.md](l1-checkpointing.md) — Snapshot/resume integration for `gonn train --resume`
- [l1-data-streaming.md](l1-data-streaming.md) — Streaming used for large datasets
- [l1-error-taxonomy.md](l1-error-taxonomy.md) — Error categories mapped to exit codes

## 1. Motivation

Reduce barrier to entry. Current GoNN requires writing a Go program to use it. A CLI lets
researchers script experiments via shell, makes the library smoke-testable from CI, and provides
a stable front-end for cross-language tooling (Python notebooks invoking via subprocess).

## 2. Constraints & Assumptions

- Single binary, distributed via `go install github.com/teratron/gonn/cmd/gonn@latest`.
- Project layout: `cmd/gonn/` — does not yet exist; created by this spec's implementation.
- No interactive UI in scope. Streaming input via stdin / files only.
- Output formats: human-readable text (default), `--json` for machine-readable.
- Routing via `switch os.Args[1]` — stdlib `flag` only, no third-party CLI frameworks (C29).
- `--precision` selects `float32` (default) or `float64` — maps to generic dispatch in `pkg/nn`.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| INV-1 (Generic Float) | CLI defaults to `float32`; `--precision=float64` switches generic dispatch. |
| INV-2 (Immutable topology) | CLI loads a frozen network config; cannot mutate during a single command. |

## 5. Detailed Design

### 5.1 Subcommands and Flags

| Subcommand | Purpose | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `gonn train` | Train a network from config + dataset | see §5.1.1 | Trained weights file + final loss |
| `gonn query` | Forward inference | see §5.1.2 | Prediction vector |
| `gonn verify` | Forward + loss without weight update | see §5.1.3 | Loss value |
| `gonn version` | Print version + Go runtime info | — | Version string |

**5.1.1 `gonn train` flags**

| Flag | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--config` | file | yes | — | Network config JSON (`l1-network-persistence`) |
| `--data` | file | yes | — | Training dataset CSV |
| `--out` | file | no | `weights.json` | Output weights file |
| `--epochs` | int | no | from config | Override max epoch count |
| `--resume` | file | no | — | Resume from checkpoint snapshot |
| `--precision` | string | no | `float32` | `float32` or `float64` |
| `--json` | bool | no | false | Machine-readable output |

**5.1.2 `gonn query` flags**

| Flag | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `--config` | file | yes | Network config JSON |
| `--weights` | file | yes | Trained weights JSON |
| `--input` | string | yes | Comma-separated input values (e.g. `"0.5,0.7"`) |
| `--precision` | string | no | `float32` or `float64` |
| `--json` | bool | no | Machine-readable output |

**5.1.3 `gonn verify` flags**

| Flag | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `--config` | file | yes | Network config JSON |
| `--weights` | file | yes | Trained weights JSON |
| `--data` | file | yes | Test dataset CSV |
| `--precision` | string | no | `float32` or `float64` |
| `--json` | bool | no | Machine-readable output |

### 5.2 Dataset Format (CSV)

- **Delimiter**: comma. **Encoding**: UTF-8. **Line endings**: LF or CRLF both accepted.
- **Header row**: optional. If the first row contains non-numeric values, it is treated as a header
  and skipped. Column count must match `inputs + outputs` dimensions from the network config.
- **Column mapping**: first `N` columns = input vector; remaining `M` columns = target vector,
  where `N` and `M` are read from the network config JSON (`inputs` / `outputs` fields).
- **Streaming threshold**: files larger than 64 MB are read via `pkg/dataset` streaming API
  (`l1-data-streaming`) to bound memory usage. Smaller files are loaded into memory entirely.
- **Error**: any row with an unparseable float or wrong column count is reported as `ErrInputData`
  (exit code 3) with row number included in the error message.

### 5.3 Exit Codes

| Code | Error category | Condition |
| :--- | :--- | :--- |
| 0 | — | Success |
| 1 | Generic | Unexpected panic or unclassified error |
| 2 | `ErrUserConfig` | Malformed config JSON, unknown flag, invalid precision |
| 3 | `ErrInputData` | Unparseable CSV row, wrong column count, missing required flag |
| 4 | `ErrTrainingFailure` | Training loop returned error (NaN loss, convergence failure) |
| 5 | `ErrIntegrity` | Checkpoint or weights file corrupt, checksum mismatch |
| 6 | `ErrIO` | File not found, permission denied, disk full |
| 7 | `ErrUnsupported` | Feature requested that is not implemented |

### 5.4 Machine-Readable Output (`--json`)

All subcommands emit a single JSON object to stdout. Shape examples:

```json
// gonn train --json
{"status":"ok","epochs":100,"final_loss":0.002,"weights_file":"weights.json"}

// gonn query --json
{"status":"ok","prediction":[0.98,0.01]}

// gonn verify --json
{"status":"ok","loss":0.0043}

// error case (any subcommand)
{"status":"error","code":3,"message":"row 42: expected 3 columns, got 2"}
```

## 6. Implementation Notes

1. Reads/writes via `l1-network-persistence` library — no separate format (Config/Weights JSON).
2. Routing: `switch os.Args[1]` → `trainCmd()` / `queryCmd()` / `verifyCmd()` / `versionCmd()`.
   Each subcommand parses its own `flag.FlagSet`.
3. Streaming: instantiate `pkg/dataset.NewCSVReader` when file stat size > 64 MB; otherwise
   `pkg/dataset.LoadCSV` (full load).
4. `--resume` passes the checkpoint path to `pkg/checkpoint.Restore` before calling `nn.Train`.
5. Exit code mapping: `errors.Is` against `pkg/utils` sentinels → `os.Exit(code)`.
6. `gonn version` prints `gonn <module_version> (go<runtime_version>)`.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[CMD]` | `cmd/gonn/` | CLI entry point — created by this spec's implementation |
| `[FACADE]` | `pkg/nn/` | Library wrapped by CLI (`nn.New`, `nn.Train`, `nn.Query`) |
| `[DATASET]` | `pkg/dataset/` | CSV reader and streaming API |
| `[PERSIST]` | `pkg/persistence/` | Config/weights JSON read-write |
| `[CHECKPOINT]` | `pkg/checkpoint/` | Resume-from-snapshot for `--resume` flag |
| `[ERRORS]` | `pkg/utils/errors.go` | Sentinel errors mapped to exit codes |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #1. |
| 0.1.0 | 2026-04-28 | Cross-refs to network-persistence and error-taxonomy added; exit-code mapping defined. Status promoted Draft → RFC. |
| 0.2.0 | 2026-05-10 | [MODIFIED] Resolved all TBDs: dataset format (CSV spec §5.2), streaming threshold (64 MB), exit codes table (§5.3), machine-readable JSON output (§5.4), complete flag tables (§5.1.1–5.1.3), related specs expanded. Status RFC → Stable. |
