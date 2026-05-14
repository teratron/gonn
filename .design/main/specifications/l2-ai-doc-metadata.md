# AI-Meta Doc-Comment Annotation

**Version:** 0.1.0
**Status:** RFC
**Layer:** implementation
**Implements:** (none — meta-convention)

## Overview

Defines the `AI-Meta:` trailing block — a structured, machine-extractable section appended to Go doc comments on exported identifiers. Its purpose is to give AI assistants and human readers constant-time orientation per symbol: what it is for, how to use it, lifecycle position, concurrency rules, error contract, related symbols, and stability promise.

The block coexists with idiomatic Go prose docstrings, renders cleanly under `go doc`, and is parseable by a stdlib-only AST walker. It uses a **closed vocabulary** of fields and **must not reference** internal SDD process artifacts (spec filenames, invariant numbers, rule numbers).

## Related Specifications

- (none — this spec is independent of L1 contracts)

## 1. Motivation

GoNN's existing doc comments are rich but free-form. AI assistants reading the codebase repeatedly re-derive the same operational facts:

- Which goroutine owns `Train`?
- What states does the network cycle through?
- Which sentinel errors does this function return?
- Which interface implementations exist and where?

The cost is paid every session and every cold cache. A short, predictable trailing block carries those facts directly on the symbol where they apply, eliminating per-session re-derivation.

A **machine-extractable** surface is the goal — not richer prose. Free-form text is already permitted by the Doc-comment Verbosity rule and does not solve the predictability problem.

## 2. Constraints & Assumptions

- **Standard library only.** Any future verifier uses `go/parser`, `go/ast`, `go/token`, `go/doc`, `go/types`. No third-party tooling.
- **Go-idiomatic.** No JavaDoc-style `@tag` syntax. Block must render correctly under `go doc` and modern IDE hover popups.
- **Process-artifact firewall.** Block content is consumer-facing documentation. It MUST NOT reference internal SDD artifacts: no `.design/...` paths, no spec filenames, no `INV-N` invariant numbers, no `C-rule` numbers. Each line is self-contained natural language.
- **Identifier names from the public API are allowed.** Sentinel error names exported from `pkg/utils/errors.go` (e.g. `ErrUserConfig`), state names from `pkg/nn/control.go`, type names — all are part of the library's public surface and may be cited freely.
- **Additive.** Existing prose is preserved. The block is appended; it does not replace existing examples or explanations.

## 3. Grammar

The block appears at the end of a doc comment, separated from preceding prose by exactly one blank doc-comment line.

```text
AI-Meta:
  - <Field>: <single-line value>
  - <Field>: <single-line value>
```

Rules:

- The literal label `AI-Meta:` introduces the block. Case-sensitive.
- Each subsequent line is `  - <Field>: <value>` — two leading spaces, hyphen, space, field name, colon, single space, value.
- Field names come from the closed vocabulary in §4. Unknown fields are a violation.
- Value is a single line. Multi-value uses comma-separated lists, never nested bullets.
- Block must be the **last** content in the doc comment (no prose after it).
- **Hard cap: 12 lines** including the `AI-Meta:` label.

## 4. Closed Vocabulary

### 4.1 Field Matrix

| Field | Public API (`pkg/nn`, `cmd/`) | Internal exported (`pkg/*`) | Unexported | Applies to |
| :--- | :--- | :--- | :--- | :--- |
| Purpose | required | required | omit | all |
| Usage | required | optional | omit | func, type, interface |
| Lifecycle | required if stateful | optional | omit | type, interface |
| Concurrency | required | required if non-trivial | omit | type, func, interface |
| Errors | required if returns `error` | required if returns `error` | omit | func |
| Related | required | optional | omit | all |
| Constraints | optional | optional | omit | type, func |
| Implementations | required | required | omit | interface only |
| Stability | required | omit (assume `Internal`) | omit | type, func, const |

### 4.2 Field Semantics

- **Purpose** — one sentence stating what the symbol is for. Prefer noun phrase for types, verb phrase for functions.
- **Usage** — one sentence on the canonical call pattern or position in a workflow.
- **Lifecycle** — state transitions involving this symbol, expressed as `<From> → <To> (<trigger>)`.
- **Concurrency** — closed enum followed by an optional clarifier after `;`.
- **Errors** — comma-separated list of sentinel names exported by the library, optionally followed by `(<short cause>)`.
- **Related** — comma-separated `[Symbol]` cross-references that godoc resolves. Names from the same package may use the bare form `[Symbol]`; cross-package references use `[pkg.Symbol]`.
- **Constraints** — natural-language behavioural rules ("Topology is immutable after Compile"). MUST NOT reference SDD artifacts.
- **Implementations** — for interfaces only: comma-separated `[Symbol]` references to all concrete types implementing this interface.
- **Stability** — closed enum.

### 4.3 Closed Enums

- `Stability` ∈ { `Stable`, `Experimental`, `Deprecated`, `Internal` }
- `Concurrency` ∈ { `Safe`, `ReadSafe`, `SingleGoroutine`, `NotSafe` } — optionally followed by `; <clarifier>`

## 5. Examples

### 5.1 Public API — type

```go
// NN is the public facade type. It embeds [network.Network] and adds the
// construction lifecycle plus the training-control state cell.
//
// Both fluent styles return *NN[T]:
//   - Builder: [NewBuilder] starts the chain in Configuring state.
//   - Options: [New] / [MustNew] run compile() implicitly and return an
//     Operational network.
//
// AI-Meta:
//   - Purpose: Public network handle returned by Builder/Options APIs.
//   - Usage: Configure topology, Compile, then Train and Query.
//   - Lifecycle: Configuring → Operational (via Compile / New / MustNew).
//   - Concurrency: ReadSafe; Train owns weights — SingleGoroutine.
//   - Related: [NewBuilder], [New], [MustNew], [network.Network].
//   - Constraints: Topology is immutable after Compile; Train mutates weights and must run alone.
//   - Stability: Stable.
type NN[T utils.Float] struct { /* ... */ }
```

### 5.2 Public API — function returning `error`

```go
// Compile finalises the staged Config and transitions the network from
// Configuring to Operational. Idempotent — repeated calls are no-ops.
//
// AI-Meta:
//   - Purpose: Lock topology and prepare the network for Train/Query.
//   - Usage: Builder API only — Options API runs Compile implicitly.
//   - Lifecycle: Configuring → Operational.
//   - Concurrency: SingleGoroutine; must complete before any Query.
//   - Errors: ErrUserConfig (missing layers, bad sizes), ErrIntegrity (axon init).
//   - Related: [NewBuilder], [NN.Train].
//   - Constraints: Idempotent; topology cannot be mutated after success.
//   - Stability: Stable.
func (n *NN[T]) Compile() (*NN[T], error) { /* ... */ }
```

### 5.3 Internal exported — interface

```go
// Nucleus is the base contract shared by all neural cells.
//
// AI-Meta:
//   - Purpose: Read-only value access for any cell type in the graph.
//   - Concurrency: ReadSafe after compile.
//   - Implementations: [cell.InputCell], [cell.HiddenCell], [cell.OutputCell], [cell.BiasCell].
//   - Related: [Neuron].
type Nucleus[T utils.Float] interface { /* ... */ }
```

### 5.4 Internal exported — error sentinel constant

```go
// ErrUserConfig flags configuration mistakes attributable to the caller.
//
// AI-Meta:
//   - Purpose: Sentinel for caller-side configuration faults.
//   - Usage: Wrap with fmt.Errorf("...: %w", ErrUserConfig); test with errors.Is.
//   - Related: [ErrIntegrity], [ErrIO].
var ErrUserConfig = errors.New("user config")
```

## 6. Anti-Patterns

The following patterns are violations.

### 6.1 SDD-artifact leakage

```go
// AI-Meta:
//   - Constraints: INV-2 (immutable topology), see .design/main/specifications/l1-neural-network-architecture.md.
```

`INV-2` and the spec path are SDD process artifacts. Replace with self-contained natural language: `Topology is immutable after Compile`.

### 6.2 Vocabulary creep

```go
// AI-Meta:
//   - Author: alice@example.com
//   - SeeAlso: docs/architecture.md
```

Both fields are outside the closed vocabulary. Authorship belongs in `git blame`; documentation pointers belong in surrounding prose if absolutely needed.

### 6.3 Multi-line value

```go
// AI-Meta:
//   - Errors:
//     - ErrUserConfig
//     - ErrIntegrity
```

Values are single-line. Use comma separation: `Errors: ErrUserConfig, ErrIntegrity`.

### 6.4 Block in the middle of the doc comment

```go
// AI-Meta:
//   - Purpose: ...
//
// Additional prose after the block.
```

The block must be the last content in the doc comment.

### 6.5 Block over the line cap

A block exceeding 12 lines (including the `AI-Meta:` label) signals over-documentation. Trim to essentials or consider whether the symbol should be split.

## 7. Verification

### 7.1 Manual (Phase 1–5)

- `go doc -all ./pkg/<package>` — visual check that the block renders as a clean labeled list.
- `go vet ./...` and `go test -race ./...` — must remain green; the block is plain doc-comment text.
- Reviewer checklist: presence of required fields per tier, no forbidden artifact references, ≤ 12 lines.

### 7.2 Automated (Phase 6, deferred)

A future `cmd/lint-aimeta` walker (stdlib-only) will enforce:

- Required fields present per tier.
- Field names within the closed vocabulary.
- Enum values within the closed sets.
- Single-line values, ≤ 12 line cap.
- `Related:` and `Implementations:` symbols resolve via `go/types`.
- No occurrences of `INV-`, `C\d+`, or `.design/` substrings inside the block (process-artifact firewall).

The walker is invoked from per-package `TestAIMetaCompliance` so `go test ./...` enforces the convention.

## 8. Rollout Phasing

Phases land as separate commits. Order is dependency-aware so `Related:` and `Implementations:` always resolve to already-annotated symbols.

| Phase | Scope | Approx. symbols |
| :--- | :--- | :--- |
| 2 | `pkg/utils` | ~15 |
| 3a | `pkg/activation` | ~22 |
| 3b | `pkg/loss` | ~30 |
| 3c | `pkg/neuron` + `pkg/layer` | ~25 |
| 4a | `pkg/network` | ~35 |
| 4b | `pkg/dataset`, `pkg/checkpoint`, `pkg/compute`, `pkg/persistence` | ~40 |
| 5 | `pkg/nn` (last — references everything above) | ~30 |
| 6 | `cmd/lint-aimeta` (deferred) | — |

Examples directory and `cmd/` (other than `lint-aimeta`) are out of scope — they are not library API surface.

## 9. Forward Compatibility

If a future Go release introduces an official structured doc format, the trailing-block design is forward-rollable: the block is still valid prose godoc. A one-pass migration script can remove every `AI-Meta:` block back to the prose-only baseline if desired.

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-07 | Initial RFC from TODO #27. Closed vocabulary, tier matrix, process-artifact firewall, phased rollout. |
