# Error Implementation

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-error-taxonomy.md

## Overview

Concrete Go realization of [l1-error-taxonomy.md](l1-error-taxonomy.md): the sentinel definitions in
`pkg/utils/errors.go`, helper constructors, the `errors.Is` / `errors.As` guarantee, and the
audit checklist for migrating existing call sites away from bare `fmt.Errorf("...")` or `errors.New`.

## Related Specifications

- [l1-error-taxonomy.md](l1-error-taxonomy.md) — Parent — invariants ERR-1..ERR-4 + 6 categories
- [RULES.md §C32](../RULES.md) — Error informativeness rule this spec operationalizes

## 1. Motivation

L1 defines the categories and contract abstractly; this spec decides Go specifics — sentinel form
(value vs type), helper signatures, log format expected by `l2-logging-strategy.md`, and migration
plan for existing codebase usage.

## 2. Constraints & Assumptions

- Stdlib only: `errors`, `fmt`.
- Sentinel = `errors.New(...)` value, not a type. Reason: smaller surface, comparable via `errors.Is`,
  no method-set requirement on call sites.
- Custom error types are reserved for cases where extra fields are required (e.g., `*ValidationError{Field, Got, Want}`).

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| ERR-1 (One return, chained) | Helpers always wrap the underlying cause via `%w`; never use `%v` for chained errors. |
| ERR-2 (Mutually exclusive categories) | Sentinels are value-comparable; helper constructors enforce one category per error. |
| ERR-3 (errors.Is routing) | `errors.Is(err, ErrUserConfig)` works because every helper output transitively wraps the matching sentinel. |
| ERR-4 (Specific messages) | Helper signatures **require** caller to pass identifying fields (size, name, value). No nullary helpers. |

## 5. Detailed Design

### 5.1 Sentinel Definitions

The taxonomy is **orthogonal**: every domain owned by a Phase-1/2/3 spec maps to exactly one
category. `ErrTrainingFailure` from v0.2.0 is dissolved (NaN/divergence → `ErrCompute`,
state-machine misuse → `ErrControl`); `ErrUnsupported` from v0.2.0 is dissolved into
`ErrUserConfig` (typo of method name) or `ErrCompute` (missing hardware feature).

| Sentinel | Domain | Owning specs |
| :--- | :--- | :--- |
| `ErrUserConfig` | API misuse before runtime — bad size, unknown method symbol | `l2-nn-facade`, layer constructors |
| `ErrInputData` | Runtime validation of caller-supplied data — bad batch shape, NaN sample | `l2-streaming-impl`, layer Forward |
| `ErrCompute` | Numeric / hardware faults inside the engine — NaN gradient, dim mismatch, no AVX2 | `l2-network-graph`, `l2-backend-cpu` |
| `ErrControl` | Training-lifecycle state-machine violations — Pause on Stopped, double-Resume | `l2-control-impl`, `l2-training-loop` |
| `ErrIntegrity` | Persisted-artifact integrity — bad checksum, schema-version mismatch | `l2-persistence-impl`, `l2-checkpointing-impl` |
| `ErrIO` | Filesystem / network — permission, EOF, disk full | `l2-persistence-impl`, `l2-checkpointing-impl` |

```go
// [REFERENCE] Public sentinels — comparable via errors.Is.
package utils

import "errors"

var (
    ErrUserConfig = errors.New("user-config")
    ErrInputData  = errors.New("input-data")
    ErrCompute    = errors.New("compute")
    ErrControl    = errors.New("control")
    ErrIntegrity  = errors.New("integrity")
    ErrIO         = errors.New("io")
)
```

### 5.2 Helper Constructors

```go
// [REFERENCE] Forced-specificity helpers.
func NewSizeError(field string, got int, constraint string) error {
    return fmt.Errorf("%s: size must be %s, got %d: %w", field, constraint, got, ErrUserConfig)
}

func NewActivationError(symbol string) error {
    return fmt.Errorf("activation %q is not registered: %w", symbol, ErrUserConfig)
}

func NewIntegrityError(what string, expected, got string) error {
    return fmt.Errorf("integrity violation in %s: expected %s, got %s: %w", what, expected, got, ErrIntegrity)
}

// ... one helper per common error site
```

Helpers are exported from `pkg/utils/errors.go` but live close to the sentinel definitions.

### 5.3 Migration Audit Checklist

The current codebase has approximately N (TBD: count after audit) bare `errors.New` / `fmt.Errorf`
sites. Migration:

1. Grep all `errors.New(` and `fmt.Errorf(` call sites in `pkg/`.
2. For each: identify the category (UserConfig / Integrity / IO / etc.).
3. Replace with the appropriate helper or wrap inline with `%w`.
4. Add a unit test: `errors.Is(err, ErrXxx)` returns true.

### 5.4 Open Questions

- <!-- TBD: include source-location field for Error-level logs (per l2-logging-strategy hint)? -->
- <!-- TBD: when do we promote a sentinel to a custom struct type (Validation case is a candidate) -->

## 6. Implementation Notes

1. New file `pkg/utils/errors.go` (does not currently exist).
2. Audit pass is a separate task — scope tracked in TASKS.md when this spec promotes.
3. Existing `pkg/utils/error.go` (singular) from gonn_old reference is not the same file; clarify in PR.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[ERRORS]` | `pkg/utils/errors.go` (new) | Sentinel + helper home |
| `[RULE]` | `.design/RULES.md#c32-error-informativeness` | Project rule this spec operationalizes |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-28 | Initial Draft — concrete Go realization of l1-error-taxonomy RFC. |
| 0.2.0 | 2026-04-28 | Status promoted Draft → RFC after parent l1-error-taxonomy reached Stable. Sentinel set + helper signatures ready for review. |
| 0.3.0 | 2026-04-29 | Sentinel set finalized to 6 orthogonal categories: ErrUserConfig, ErrInputData, ErrCompute, ErrControl, ErrIntegrity, ErrIO. ErrTrainingFailure and ErrUnsupported dissolved into existing categories to avoid catch-all routing. |
| 1.0.0 | 2026-04-30 | Promoted RFC → Stable. Validated by Phase-1 implementation: pkg/utils/errors.go ships with 100% test coverage, race-detector clean, C32 forbidden-phrase guard active. Spec text and code in lock-step. |
