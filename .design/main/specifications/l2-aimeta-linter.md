# AI-Meta Linter — `cmd/lint-aimeta`

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-neural-network-architecture.md

## Overview

Concrete realization of the **automated verification** path described in
[l2-ai-doc-metadata.md](l2-ai-doc-metadata.md) §7.2 — a stdlib-only AST walker delivered as
`cmd/lint-aimeta` (standalone CLI) plus a thin import-side adapter under `pkg/aimeta` so the same
checker can be invoked from per-package `TestAIMetaCompliance` test functions. The linter enforces
the closed vocabulary, tier matrix, single-line-value rule, 12-line cap, and process-artifact
firewall defined in `l2-ai-doc-metadata.md` §3 and §4.

The scope is intentionally narrow: structural validation of `AI-Meta:` blocks on exported
identifiers. Prose quality, English grammar, and semantic correctness are out of scope.

## Related Specifications

- [l2-ai-doc-metadata.md](l2-ai-doc-metadata.md) — Convention being enforced; §4 vocabulary is the linter's spec
- [l2-cli-client.md](l2-cli-client.md) — Sibling `cmd/` binary; shares CLI conventions (exit codes, `--json`, flag style)
- [l2-errors-impl.md](l2-errors-impl.md) — Error sentinels used by the linter must wrap `ErrUserConfig` for caller-side faults
- [l1-error-taxonomy.md](l1-error-taxonomy.md) — Category sentinel taxonomy

## 1. Motivation

`l2-ai-doc-metadata.md` defines a structured trailing block in doc comments. Without automated
enforcement, the convention drifts: fields outside the closed vocabulary creep in (e.g., `Author:`,
`SeeAlso:`), values become multi-line, blocks exceed the 12-line cap, and SDD-artifact strings
(`INV-`, `C\d+`, `.design/`) leak into consumer-facing docs.

Manual review catches obvious cases but does not scale beyond a handful of files. Per
`l2-ai-doc-metadata.md` §7.2 a future walker will close this gap; this spec is that delivery.

The linter is **standalone** so contributors and CI can run it without booting any other GoNN
package; it is also **importable** via `pkg/aimeta` so per-package `TestAIMetaCompliance` test
functions invoke the same code path and fail the test suite on violations.

## 2. Constraints & Assumptions

- **Stdlib only (C29)**: `go/parser`, `go/ast`, `go/token`, `go/doc`, `go/types`, `go/build`. No third-party tooling.
- **Read-only**: linter never writes source files. Auto-fix is explicitly out of scope; violation reports point at the offending line.
- **CI-safe**: exit code 0 (clean) / 1 (violations found) / 2 (internal error) / 3 (parse error) — same pattern as `cmd/gonn` (l2-cli-client §5.3).
- **Tier-aware**: visibility tier is inferred from package path — `pkg/nn/...`, `cmd/...` → Public; `pkg/<other>/...` (exported symbols) → Internal exported; unexported identifiers are skipped.
- **Resolver opt-in**: `Related:` and `Implementations:` symbol resolution via `go/types` is enabled by `--resolve` flag. Default mode is grammar-only (faster, no module compilation).
- **Process-artifact firewall**: forbidden substrings inside the block — `INV-`, `C\d+` (regex), `.design/`. Detected via byte-level scan of the block text after AST extraction.

## 4. Invariant Compliance

> This L2 implements the convention defined in `l2-ai-doc-metadata.md`. Each row maps a §3/§4
> rule to a linter check.

| Convention rule (source) | Implementation |
| :--- | :--- |
| §3 — `AI-Meta:` literal label, case-sensitive | `strings.HasPrefix(line, "AI-Meta:")` byte check |
| §3 — two-space `  - <Field>: <value>` indentation | Regex `^  - [A-Z][a-zA-Z]+: .+$` per line |
| §3 — block is the LAST content in the doc comment | AST: last `*ast.Comment` group on the symbol; reject if prose follows |
| §3 — hard cap 12 lines including label | Count `\n` in block text; fail if > 11 newlines |
| §4.1 — closed Field vocabulary | Match field name against `var AllowedFields = []string{"Purpose","Usage","Lifecycle","Concurrency","Errors","Related","Constraints","Implementations","Stability"}` |
| §4.1 — tier-required field presence (Public > Internal > Unexported) | Tier inferred from package path + identifier export; required-field set checked per row of §4.1 matrix |
| §4.2 — single-line value | Block split by `\n`; each non-label line must produce exactly one `field: value` pair |
| §4.3 — `Stability` ∈ `{Stable, Experimental, Deprecated, Internal}` | Enum check on value (trim leading/trailing whitespace) |
| §4.3 — `Concurrency` ∈ `{Safe, ReadSafe, SingleGoroutine, NotSafe}` (optional `; clarifier`) | Split value on `;`; first token must be in enum |
| §4 — `Related: [Symbol]` / `[pkg.Symbol]` references resolve via `go/types` | Opt-in `--resolve` flag; uses `golang.org/x/tools/go/packages`... — REJECTED, stdlib only → use `go/build` + `go/types` directly |
| §6.1 — no `INV-`, `C\d+`, `.design/` substrings | Byte-level scan after AST extraction; report exact line+col |
| §6.3 — multi-line value forbidden | Same as §4.2 single-line check |
| §6.4 — no prose after the block | Same as §3 last-content check |

## 5. Detailed Design

### 5.1 Package Layout

```text
pkg/aimeta/
├── lint.go         // Public API: Check(pkgPath string, opts Options) ([]Violation, error)
├── ast.go          // ExtractBlock(*ast.CommentGroup) (Block, bool)
├── grammar.go      // ParseBlock(text string) (Block, []Violation)
├── vocab.go        // AllowedFields, Stability enum, Concurrency enum, TierRequiredFields
├── resolver.go     // ResolveSymbols(block Block, pkg *types.Package) []Violation (opt-in)
├── violation.go    // Violation type with token.Pos + Rule code
└── lint_test.go    // Golden-file table tests

cmd/lint-aimeta/
├── main.go         // CLI entry point; dispatch to pkg/aimeta
├── output.go       // Text and JSON formatters
├── exitcode.go     // 0/1/2/3 contract
└── main_test.go    // End-to-end CLI tests
```

### 5.2 Public API — `pkg/aimeta`

```go
// [REFERENCE] In pkg/aimeta/lint.go.
package aimeta

// Violation reports a single rule failure with source location.
type Violation struct {
    Pos     token.Position // file:line:col of the offending line
    Symbol  string         // identifier name; empty if violation is at block level
    Rule    string         // e.g. "VOCAB", "TIER", "CAP", "ARTIFACT", "ENUM"
    Message string         // human-readable description
}

// Options tune linter behaviour.
type Options struct {
    Resolve         bool // enable go/types symbol resolution for Related / Implementations
    IncludeInternal bool // check internal-exported tier as well as public (default: true)
}

// Check walks pkgPath, parses each .go file, extracts AI-Meta blocks from
// doc comments on exported identifiers, validates against the convention,
// and returns all violations found. Returns a non-nil error only on parse
// or filesystem failures — rule violations are returned via the slice.
func Check(pkgPath string, opts Options) ([]Violation, error) {
    // 1. Walk pkgPath recursively, filter to .go files (skip *_test.go unless --tests).
    // 2. For each file: parser.ParseFile, iterate Decls, extract Doc *ast.CommentGroup
    //    for exported identifiers.
    // 3. For each comment group: locate `AI-Meta:` label, extract block lines.
    // 4. Run grammar checks (ParseBlock).
    // 5. Run tier checks based on package path + identifier export.
    // 6. Run artifact-firewall byte scan.
    // 7. If opts.Resolve: build *types.Package via go/build + go/types; validate
    //    Related: and Implementations: symbol references.
    // 8. Return aggregated violations.
}
```

### 5.3 CLI Entry — `cmd/lint-aimeta`

```text
Usage: lint-aimeta [flags] <package-pattern>...

Flags:
  --resolve       Enable symbol resolution for Related / Implementations (default: false)
  --json          Emit violations as newline-delimited JSON instead of text
  --include-tests Lint *_test.go files as well (default: false)
  --no-internal   Skip internal-exported tier; only check Public tier (pkg/nn, cmd)

Package patterns:
  ./...                          recursive from current dir
  github.com/teratron/gonn/...   absolute, recursive
  pkg/nn pkg/network             explicit list
```

Default invocation in CI: `lint-aimeta ./pkg/... ./cmd/...` (no `--resolve` for speed; symbol
resolution is performed by per-package `TestAIMetaCompliance` tests instead, which already have
the full package loaded).

### 5.4 Per-Package Test Hook

```go
// [REFERENCE] In pkg/nn/aimeta_test.go (example — one per package per §8 Rollout Phasing).
package nn_test

import (
    "testing"

    "github.com/teratron/gonn/pkg/aimeta"
)

func TestAIMetaCompliance(t *testing.T) {
    violations, err := aimeta.Check(".", aimeta.Options{
        Resolve:         true, // package already loaded by go test
        IncludeInternal: true,
    })
    if err != nil {
        t.Fatalf("aimeta.Check: %v", err)
    }
    for _, v := range violations {
        t.Errorf("%s: %s [%s] %s", v.Pos, v.Symbol, v.Rule, v.Message)
    }
}
```

This pattern is added by each rollout phase (Phase 2-5 per `l2-ai-doc-metadata.md` §8) as that
package's AI-Meta annotations land.

### 5.5 Exit Code Contract

| Code | Meaning |
| :--- | :--- |
| 0 | Clean — no violations |
| 1 | Violations found — see output |
| 2 | Internal error (panic, unexpected I/O) |
| 3 | Parse error — package(s) failed to compile or one or more files unparseable |

Matches the `cmd/gonn` taxonomy from `l2-cli-client.md` §5.3 with overlap on 0-3; codes 4-7 are
reserved for future linter expansion.

### 5.6 Rule Codes (Violation.Rule)

| Code | Source | Meaning |
| :--- | :--- | :--- |
| `LABEL` | §3 | Missing or malformed `AI-Meta:` label |
| `INDENT` | §3 | Wrong indentation / hyphen-space prefix |
| `CAP` | §3 | Block exceeds 12 lines |
| `LAST` | §3 + §6.4 | Block is not the last content in the doc comment |
| `VOCAB` | §4.1 + §6.2 | Field name outside closed vocabulary |
| `TIER` | §4.1 | Required field missing for symbol's tier |
| `MULTI` | §4.2 + §6.3 | Multi-line value detected |
| `ENUM` | §4.3 | `Stability` / `Concurrency` value outside enum |
| `ARTIFACT` | §6.1 | SDD-artifact substring (`INV-`, `C\d+`, `.design/`) inside block |
| `RESOLVE` | §4.2 | `Related:` / `Implementations:` symbol cannot be resolved (only with `--resolve`) |

## 6. Implementation Notes

1. **Phase A** — `pkg/aimeta` skeleton + grammar checks (LABEL, INDENT, CAP, LAST, VOCAB, TIER, MULTI, ENUM, ARTIFACT). No resolver. Unit tests via golden-file fixtures in `pkg/aimeta/testdata/`.
2. **Phase B** — `cmd/lint-aimeta` CLI + text + JSON output + exit code contract. End-to-end smoke test in `cmd/lint-aimeta/main_test.go`.
3. **Phase C** — Resolver (`--resolve` flag) backed by `go/build` + `go/types`. Adds the `RESOLVE` rule code. Lazy — only loads packages when flag is set.
4. **Phase D** — First per-package `TestAIMetaCompliance` hook in `pkg/utils/` (smallest package, simplest annotations). Becomes the template for rollout phases 2-5 of `l2-ai-doc-metadata.md` §8.

Phase D landing is the **promotion gate** from v0.1.0 (this spec) to v1.0.0 — once a real package
validates clean end-to-end, the convention has tooling parity with manual review.

## 7. Drawbacks & Alternatives

- **Alternative: third-party linter framework (`x/tools/go/analysis`)** — rejected; violates C29 stdlib-only and adds a transitive `golang.org/x/tools` dependency that the entire library otherwise avoids.
- **Alternative: comment-only linter via `regexp` without `go/ast`** — rejected; needs AST to distinguish doc comments on exported identifiers from in-function comments, and to compute identifier tier from export status.
- **Auto-fix mode** — explicitly out of scope. AI-Meta block content is human-curated semantic data; mechanical insertion of `Purpose:` lines would produce empty placeholders. Surface violations, let humans write the value.
- **Cyclomatic-complexity-style severity grading** — out of scope. All violations are equal in `lint-aimeta`; the CI policy decides whether any violation fails the build.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[CONVENTION]` | `.design/main/specifications/l2-ai-doc-metadata.md` (§3-§4) | Grammar + closed vocabulary — the spec being enforced |
| `[VERIFY-7-2]` | `.design/main/specifications/l2-ai-doc-metadata.md` (§7.2) | Original linter contract — preserve scope verbatim |
| `[ROLLOUT-§8]` | `.design/main/specifications/l2-ai-doc-metadata.md` (§8) | Package order for `TestAIMetaCompliance` hook delivery |
| `[CLI-CONV]` | `.design/main/specifications/l2-cli-client.md` (§5.3) | Exit code contract template — overlapping codes 0-3 |
| `[ERR-PKG]` | `pkg/utils/errors.go` | `ErrUserConfig` sentinel — wrap on caller-side faults |
| `[ANTI-§6]` | `.design/main/specifications/l2-ai-doc-metadata.md` (§6) | Negative examples — golden-file fixtures in `pkg/aimeta/testdata/` mirror this section |

<!-- Downstream agent instruction: §5.1 package layout is the file blueprint. §5.6 rule codes are
     the stable vocabulary — adding a new rule code requires a minor version bump and a §4 row. -->

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-18 | Initial spec authored via `/magic-spec` Spark 3b. Stdlib-only AST walker delivered as `cmd/lint-aimeta` (CLI) + `pkg/aimeta` (importable). 10 rule codes (LABEL/INDENT/CAP/LAST/VOCAB/TIER/MULTI/ENUM/ARTIFACT/RESOLVE), 4-phase implementation plan. Promoted Draft → Stable via Trust Mode (MVC: Overview + §4 Invariant Compliance + §5 Detailed Design; no RULES conflicts; no cycles). |
