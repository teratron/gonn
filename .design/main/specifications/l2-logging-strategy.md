# Logging Strategy

**Version:** 0.2.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-observability-protocol.md

## Overview

Specifies how GoNN emits structured logs at multiple levels using the standard library's `log/slog`.
Aligns logging with the read-only observability contract (`l1-observability-protocol.md`) so the same
events surfaced to GUIs are also logged to a configurable sink. Targets production-grade verbosity
levels: Trace / Debug / Info / Warn / Error.

## Related Specifications

- [l1-observability-protocol.md](l1-observability-protocol.md) — Parent
- [l1-error-taxonomy.md](l1-error-taxonomy.md) — Error categories the logger emits

## 1. Motivation

The current `pkg/utils/Logger` is opaque — no documented levels, no structured fields, no contract
on what fires when. Production users want predictable, filterable, machine-parseable logs they can
ship to ELK / Loki / Cloud Logging. `log/slog` (Go 1.21+) is the standard answer.

## 2. Constraints & Assumptions

- Standard library only (`log/slog`). No `zap`, `zerolog`, etc. (per `C29`).
- Default sink: stderr, level Info, text format. Configurable via `WithLogger(*slog.Logger)`.
- Log levels follow slog convention: `LevelDebug=-4`, `LevelInfo=0`, `LevelWarn=4`, `LevelError=8`.
  Trace is custom: `LevelTrace = -8`.
- Logging must be **best-effort**: a logger error never crashes training. Use a fallback discard sink.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| OBS-1 (Snapshots) | Logged events carry value copies, not live pointers. |
| OBS-3 (Zero overhead when off) | If level filters out the event, no field allocation occurs (slog's `Enabled()` gate). |
| OBS-4 (Versioned) | Schema version embedded as the `lib_version` field on every event. |

## 5. Detailed Design

### 5.1 Level Conventions

| Level | When to emit | Example |
| :--- | :--- | :--- |
| Trace | Per-iteration mechanics (forward/backward steps) | `trace iter=42 forward_ms=0.3` |
| Debug | Per-epoch progress, diagnostics | `debug epoch=10 loss=0.05` |
| Info | Lifecycle events: train start/stop, snapshot, network compiled | `info training started input=2 output=1` |
| Warn | Recoverable concerns: soft validation warnings, retry events | `warn loss_diverging delta=+0.05` |
| Error | Unrecoverable: aborted training, IO failure | `error training_failed err="..."` |

### 5.2 Required Fields per Event

Every log event includes:

- `lib_version` — GoNN version string.
- `network_id` — short hash of the compiled topology (stable identifier).
- `iter` — current iteration counter when in training.
- `loss` — current loss when in training.

Plus event-specific fields documented per call-site.

### 5.3 Open Questions

- <!-- TBD: include source-file fields (slog's AddSource)? probably yes for Error level only -->
- <!-- TBD: log sampling for Trace level under heavy load — every Nth iteration? -->

### 5.4 Package Structure Changes

```plaintext
pkg/utils/
├── logger.go         # Replace opaque Logger with slog-based thin wrapper
└── logger_test.go    # Tests for LevelTrace constant, required fields
```

`pkg/nn/` option wiring:

```plaintext
pkg/nn/
├── options.go   # WithLogger[T](l *slog.Logger) option
└── config.go    # Logger *slog.Logger field on nnConfig
```

### 5.5 Logger Struct

```go
// [REFERENCE]
const LevelTrace = slog.Level(-8)

// goLogger wraps *slog.Logger and injects required fields on every call.
type goLogger struct {
    l          *slog.Logger
    libVersion string
    networkID  string
}

func newGoLogger(l *slog.Logger, version, id string) *goLogger
func (g *goLogger) Trace(msg string, args ...any)
func (g *goLogger) Debug(msg string, args ...any)
func (g *goLogger) Info(msg string, args ...any)
func (g *goLogger) Warn(msg string, args ...any)
func (g *goLogger) Error(msg string, args ...any)
```

Required fields (`lib_version`, `network_id`, `iter`, `loss`) are injected via `slog.With` at
construction time for the first two, and passed as positional args at call-sites for `iter`/`loss`
(which change per iteration).

Default handler: `slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})`.
Fallback: if the provided logger is nil, a discard handler is used (no-op, no crash).

## 6. Implementation Notes

1. Add `LevelTrace` constant to `pkg/utils/logger.go`.
2. Replace opaque `Logger` type with `goLogger` wrapping `*slog.Logger`.
3. Add `WithLogger[T](*slog.Logger)` to `pkg/nn/options.go` and `builder.go`.
4. Add call-sites in `pkg/nn/train.go` (iteration Trace, epoch Debug, start/stop Info).
5. Address the two TBD questions in §5.3: add source fields on Error level only;
   defer Trace sampling to a follow-up minor.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[LOGGER]` | `pkg/utils/logger.go` | Logger implementation — slog wrapper to replace current shim |
| `[NN-OPT]` | `pkg/nn/options.go` | `WithLogger` option wiring |
| `[TRAIN]` | `pkg/nn/train.go` | Primary call-site for Trace/Debug/Info log events |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #9. |
| 0.1.0 | 2026-05-07 | [Pre-Plan] Trust Mode promoted Draft → Stable. MVC satisfied. |
| 0.2.0 | 2026-05-10 | Added §5.4 package structure, §5.5 goLogger struct with LevelTrace, resolved TBD on source fields. Updated canonical references. |
