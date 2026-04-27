# CLI Client for GoNN

**Version:** 0.1.0
**Status:** Draft
**Layer:** implementation
**Implements:** l1-neural-network-architecture.md

## Overview

A standalone command-line client `gonn` that wraps the library's public API for non-Go users and CI
pipelines. Provides train / query / verify subcommands operating on JSON-described networks and tabular
datasets, without requiring users to compile their own Go binary.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent
- [l2-nn-facade.md](l2-nn-facade.md) — Library API the CLI delegates to
- [l1-checkpointing.md](l1-checkpointing.md) — Snapshot/resume integration <!-- TBD: when checkpointing lands -->

## 1. Motivation

Reduce barrier to entry. Current GoNN requires writing a Go program to use it. A CLI lets researchers
script experiments via shell, makes the library smoke-testable from CI, and provides a stable
front-end for cross-language tooling (Python notebooks invoking via subprocess).

## 2. Constraints & Assumptions

- Single binary, distributed via `go install github.com/teratron/gonn/cmd/gonn@latest`.
- Project layout: `cmd/gonn/` — does not yet exist; created by this spec's implementation.
- No interactive UI in scope. Streaming input via stdin / files only.
- Output formats: human-readable text (default), `--json` for machine-readable.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| INV-1 (Generic Float) | CLI defaults to `float32`; `--precision=float64` switches generic dispatch. |
| INV-2 (Immutable topology) | CLI loads a frozen network config; cannot mutate during a single command. |

## 5. Detailed Design

### 5.1 Subcommands

| Subcommand | Purpose | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `gonn train` | Train a network from config + dataset | `--config nn.json --data data.csv` | Trained weights file `--out weights.json` + final loss |
| `gonn query` | Forward inference | `--config nn.json --weights weights.json --input "0.5,0.7"` | Prediction vector |
| `gonn verify` | Forward + loss without weight update | `--config nn.json --weights weights.json --data test.csv` | Loss value |
| `gonn version` | Print version + Go runtime info | — | Version string |

### 5.2 Open Questions

- <!-- TBD: dataset format — start with CSV; consider Parquet / TFRecord adapters later -->
- <!-- TBD: integration with `l1-data-streaming.md` for large datasets -->
- <!-- TBD: exit codes contract — 0 success, non-zero per error category from l1-error-taxonomy -->

## 6. Implementation Notes

1. Gated on `l1-network-persistence.md` (planned) — config/weights JSON schema.
2. Build with `cobra` or stdlib `flag` only; per `C29 — Zero External Dependencies`, default to stdlib `flag`.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[CMD]` | `cmd/gonn/` | New directory — CLI entry point |
| `[FACADE]` | `pkg/nn/` | Library wrapped by CLI |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #1. |
