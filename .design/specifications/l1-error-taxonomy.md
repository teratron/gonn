# Error Taxonomy

**Version:** 0.1.0
**Status:** Draft
**Layer:** concept

## Overview

Defines the canonical error categories returned by GoNN, the contract on error wrapping, and the
mapping from error category to user remedy. Every public function in the library returns errors that
fit one of these categories, enabling callers to route by `errors.Is` / `errors.As` rather than
parsing strings.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent
- [l2-logging-strategy.md](l2-logging-strategy.md) — Logger emits events by category at appropriate level

## 1. Motivation

Today error handling is ad-hoc: bare `fmt.Errorf` calls, inconsistent wrapping, opaque messages.
Library users cannot programmatically distinguish "user passed a bad config" from "training diverged"
from "snapshot file corrupt" — they get strings. A taxonomy makes errors **diagnosable, actionable,
and machine-routable**.

## 2. Constraints & Assumptions

- Errors implement standard `error` interface; categories use `errors.Is` via sentinel values or
  category-specific types.
- Error messages are **human-actionable** — they say what went wrong, where, and (when possible) how
  to fix it.
- Wrapping uses `fmt.Errorf("...: %w", err)` per project rule.
- No panic in library code except `MustXxx` constructors. All other errors are returned values.

## 3. Core Invariants

- **ERR-1**: Every public function returns at most one error. Wrapping preserves the chain so
  `errors.Unwrap` recovers root cause.
- **ERR-2**: Every error belongs to **exactly one** category. Categories are mutually exclusive.
- **ERR-3**: Category root sentinel (`ErrUserConfig`, `ErrIntegrity`, etc.) is comparable via
  `errors.Is`. Specific errors wrap a category sentinel: `fmt.Errorf("input size 0: %w", ErrUserConfig)`.
- **ERR-4**: Error messages are **specific**. Forbidden phrases: "something went wrong", "internal
  error", "unknown error". Mandatory: identify the offending value or location when known.

## 5. Detailed Design

### 5.1 Categories

| Sentinel | Meaning | Example | Recommended caller action |
| :--- | :--- | :--- | :--- |
| `ErrUserConfig` | Caller passed invalid configuration | `Input(0)`, unknown activation symbol | Fix input, retry |
| `ErrInputData` | Runtime input data mismatch | Sample length differs from `inputSize` | Validate dataset, retry |
| `ErrTrainingFailure` | Training diverged or hit max iterations without convergence | NaN loss | Adjust hyperparameters |
| `ErrIntegrity` | Internal state corrupted or invariant violated | Snapshot checksum mismatch | Stop, file bug |
| `ErrIO` | Filesystem / network failure | Snapshot write failed | Retry with backoff |
| `ErrUnsupported` | Operation not implemented in current build | GPU backend on CPU-only build | Use alternative |

### 5.2 Wrapping Example (Pseudo-code)

```text
if size == 0:
    return fmt.Errorf("Input(): size must be positive, got 0: %w", ErrUserConfig)
```

### 5.3 Open Questions

- <!-- TBD: stack-trace inclusion (debug.Stack on Error level) -->
- <!-- TBD: error codes for CLI exit-code mapping (gonn cli) -->
- <!-- TBD: localization — out of scope for now (English-only per project rule §1.1) -->

## 6. Implementation Notes

1. Define sentinels in `pkg/utils/errors.go`.
2. Audit every existing `errors.New` / `fmt.Errorf` and route into a category.
3. Add `errors.Is` examples to public docstrings.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[ERRORS]` | `pkg/utils/errors.go` (new) | Sentinel definitions |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #8 (concept side; complementary rule is C32). |
