# Project Specification Rules

**Version:** 1.3.0
**Status:** Active

## Overview

Constitution of the specification system for this project.
Read by the agent before every operation. Updated only via explicit triggers.

## 1. Naming Conventions

- Spec files must include a layer prefix (e.g., `l1-`, `l2-`), followed by lowercase kebab-case: `l1-api.md`, `l2-database-schema.md`.
- System files use uppercase: `INDEX.md`, `RULES.md`.
- Section names within specs are title-cased.

## 2. Status Rules

- **Draft → RFC**: all required sections filled, ready for review.
- **RFC → Stable**: reviewed, approved, no open questions.
- **RFC → Draft**: needs rework or revision affecting ≥1 core section.
- **Stable → RFC**: substantive amendment (minor/major bump) requires re-review.
- **Any → Deprecated**: explicitly superseded; replacement must be named.

## 3. Versioning Rules

- `patch` (0.0.X): typo fixes, clarifications — no structural change.
- `minor` (0.X.0): new section added or existing section extended.
- `major` (X.0.0): structural restructure or scope change.

## 4. Formatting Rules

- Use `plaintext` blocks for all directory trees.
- Use `mermaid` blocks for all flow and architecture diagrams.
- Do not use other diagram formats.

## 5. Content Rules

- No implementation code (no Rust, JS, Python, SQL, etc.).
- Pseudo-code and logic flows are permitted.
- Every spec must have: Overview, Motivation, Document History.

## 6. Relations Rules

- Every spec that depends on another must declare it in `Related Specifications`.
- Cross-file content duplication is not permitted — use a link instead.
- Circular dependencies must be flagged and resolved.

## 7. Project Conventions

### C1 — `.magic/` Engine Safety

`.magic/` is the active SDD engine. Any modification must follow this protocol:

1. **Read first** — open and fully read every file that will be affected.
2. **Analyse impact** — trace how the changed file is referenced by other engine files and workflow wrappers.
3. **Verify continuity** — confirm that after the change all workflows remain fully functional.
4. **Never edit blindly** — if the scope of impact is unclear, stop and ask before proceeding.
5. **Document the change** — record modifications in the relevant spec and commit message.
6. **Atomic Update** — apply changes simultaneously across all related files (scripts, workflows, and documentation) to maintain full engine consistency.
7. **No-Change, No-Bump** — NEVER trigger a version bump (C14) if no physical files in `.magic/` were modified (e.g., during dry runs or purely cognitive tasks).

### C2 — Workflow Minimalism

Limit the SDD workflow to the core command set to maximize automation and minimize cognitive overhead. Do not introduce new workflow commands unless strictly necessary and explicitly authorized as a C2 exception.

### C3 — Parallel Task Execution Mode

Task execution defaults to **Parallel mode**. A Manager Agent coordinates execution, reads status, unblocks tracks, and escalates conflicts. Tasks with no shared constraints are implemented in parallel tracks.

### C4 — Automate User Story Priorities

Skip the user story priority prompt. The agent must automatically assign default priorities (P2) to User Stories during task generation to maximize automation and avoid interrupting the user.

### C6 — Selective Planning

During plan updates, specs are handled by their status:

- **Draft specs**: automatically moved to `## Backlog` in `PLAN.md` without user input.
- **RFC specs**: surfaced to user with a recommendation to backlog until Stable.
- **Stable specs**: agent asks which ones to pull into the active plan. All others go to Backlog.
- **Orphaned specs** (in INDEX.md but absent from both plan and backlog): flagged as critical blockers.

### C7 — Universal Script Executor

All automation scripts must be invoked via the cross-platform executor:
`node .magic/scripts/executor.js <script-name> [args]`

Direct calls to `.sh` or `.ps1` scripts are not permitted in workflow instructions. The executor detects the OS and delegates to the platform-matching implementation.

### C8 — Phase Archival

On phase completion, the per-phase task file is moved from `$DESIGN_DIR/tasks/` to `$DESIGN_DIR/archives/tasks/`. The link in `TASKS.md` is updated to point to the archive location. This keeps the active workspace small while preserving full history.

### C9 — Zero-Prompt Automation (Trust Mode)

Once the user provides high-level intent (ideation), the agent is authorized to proceed through the entire lifecycle (Draft → RFC → Stable → Plan → Task → Run) without further confirmation prompts, provided the logic is clear and non-conflicting. Silent operations include: status auto-promotion, planning, retrospective Level 1, changelog Level 1, and CONTEXT.md regeneration. Critical exceptions requiring explicit user approval:

1. **Changelog Level 2** (external release artifacts).
2. **Destructive Actions** (deleting files or specifications).
3. **Ambiguous Triggers** (where >1 architectural path exists).

### C10 — Task Architecture & Status Truth

Logic and progress tracking are distributed between two primary files to ensure clarity and automation:

1. **`PLAN.md` (Strategic)**: High-level overview of **Phase → Specification**. Each specification has a single checkbox representing its aggregate implementation status.
2. **`TASKS.md` (Tactical)**: The master execution ledger. Contains a concise **Phase Checklist** (items prefixed with unique `[T-XXXX]` IDs) followed by detailed task blocks.

All execution progress (`[x]`, `[/]`, etc.) must be recorded in the `TASKS.md` checklist first. `PLAN.md` is updated only when a specification or phase is fully completed.

### C11 — Simulation Workflow (C2 Exception)

`magic.simulate` is explicitly authorized as a developer-facing tool for engine validation and regression testing. It is a one-time exception to C2. Not intended for use in regular project workflows.

### C12 — Quarantine Cascade

If a Layer 1 (Concept) specification loses its `Stable` status or is removed, all dependent Layer 2/3 (Implementation) specifications must automatically and transparently be treated as demoted to `RFC` or moved to the Backlog by the Task workflow. The system must quarantine dependent specifications to prevent "orphaned" task scheduling without requiring manual status edits for every child in `INDEX.md`.

**C12.1 — Stabilization Exception**: Tasks explicitly intended to stabilize or fix mismatches to regain `Stable` status for the parent may bypass this quarantine.

### C13 — Agent Cognitive Discipline

All AI agents operating within the Magic SDD framework must adhere to strict cognitive discipline to prevent hallucinations and silent failures:

1. **Primary Source Principle**: Always read original `.magic/` and `.design/` files. Never rely on cached memory or interpretive assumptions.
2. **Anti-Truncation**: Execute checklists and multi-step processes literally. Do not skip, merge, or summarize steps.
3. **Zero Assumptions**: If an instruction is absent or ambiguous, halt and ask for clarification. Do not invent missing steps or scripts.
4. **Mandatory Self-Verification**: Cross-reference actions against original instructions before finalizing any task or presenting a completion checklist.
5. **Anti-Hallucination Audit**: All architectural conclusions, problem reports, and proposed changes must be directly traceable to specific statements within project specifications or engine rules.

### C14 — Engine Versioning Protocol

To ensure accurate engine state tracking and reliable updates, any modification to the core engine/kernel files (anything inside the `.magic/` directory, including workflows and templates) MUST be accompanied by an automated engine metadata update: `node .magic/scripts/executor.js update-engine-meta --workflow {workflow}`.

1. **Scope**: Applies to all `.md` workflows, `scripts/`, `templates/`, and `config.json` inside the engine directory.
2. **Automation**: This command automatically increments the patch version in `.magic/.version`, updates the relevant history file in `.magic/history/`, and regenerates `.magic/.checksums`. **Smart History**: Redundant automated entries are skipped if the last entry matches.
3. **Exclusion**: Modifications to `.design/` files (project content) do NOT trigger an engine version bump; they trigger project manifest bumps instead.
4. **Synchronization**: The version in `.magic/.version` should stay aligned with the latest meaningful change to the engine's functional logic.
5. **Simulation Exemption**: Purely cognitive simulations, dry runs, or audit tasks that do not modify files MUST NOT trigger a C14 version bump to avoid metadata noise.

### C15 — Workspace Scope Isolation

When operating in a workspace with a defined scope (via `.design/workspace.json`), the agent MUST restrict all analysis and file operations to the directories specified in the scope. All other project directories are treated as out-of-scope to ensure logical isolation and prevent context leakage or accidental modification of unrelated modules.

### C16 — Micro-spec Convention

For minor features, simple bugfixes, or changes expected to be under 50 lines of documentation, the agent is authorized to use the lightweight `.magic/templates/micro-spec.md` instead of the full specification template. If a Micro-spec exceeds 50 lines or architectural complexity increases, it MUST be promoted to the full Standard template.

### C17 — Adapter Registry

All new IDE/Agent adapters must be registered in `installers/adapters.json`. This registry is the single source of truth for installer deployment paths and marker files.

### C18 — Payload Security

The installers (Node/Python) must verify payload integrity (checksums) and prevent Path Traversal attacks during extraction. Deployment must be atomic to prevent partial engine states.

### C19 — Cross-Env CLI Parity

Node and Python installers must maintain strict CLI parity. Every command-line flag (e.g., `--yes`, `--update`, `--check`) must behave identically across both implementations to ensure a consistent user experience.

### C20 — Auto-Heal Recovery

The engine must proactively identify and repair its own metadata. If `executor.js` detects missing history files or corrupted checksums during non-critical operations, it should attempt to "Auto-Heal" (restore defaults or regenerate) before Proceeding or Halting.

### C21 — Project Ventilation (Analyze)

The command `/magic.analyze` (or `Analyze project`) triggers "Project Ventilation": a deep scan that treats the current codebase as the source of truth and compares it against `INDEX.md` and `RULES.md`. It must identify:

- **Registry Drift**: Specs in INDEX but missing on disk.
- **Coverage Gaps**: Code folders without corresponding specs.
- **Rule Violations**: Code patterns that contradict `RULES.md §7` (both global and workspace tiers).
- **Integrity Issues**: Mismatched checksums in `.magic/`.

### C22 — Workspace Rule Inheritance

Each workspace may maintain a local `RULES.md` at `.design/{workspace}/RULES.md`. These files:

1. Contain only workspace-specific §7 conventions, identified as `WC1`, `WC2`, … (workspace convention).
2. Inherit all §1-6 universal rules and global §7 conventions from `.design/RULES.md` — no re-declaration needed.
3. Must not contradict the global constitution (Constitutional Guard applies equally).
4. Are created on demand by `magic.rule` when the first workspace-scoped rule is requested.
5. Version independently from the global `RULES.md`.

### C23 — Context Economy & Validation Caching

To minimize redundant resource usage and improve performance, the agent may optimize `check-prerequisites` calls within a single task lifecycle:

1. **Turn-Aware Caching**: If `check-prerequisites` returned `ok: true` earlier in the current conversation turn or the immediately preceding turn, and the agent has NOT modified any files in `.magic/` or `.design/` since that check, the agent is authorized to skip the physical script execution and rely on the known "Clean State".
2. **External Drift Guard**: If >5 minutes have passed since the last check, the context window has been compacted, or the user has performed manual file operations (e.g. `git pull`, manual edits in terminal), the agent MUST perform a fresh `check-prerequisites` call.
3. **Halt Persistence**: If the previous check returned an error or warning (e.g. `checksums_mismatch`), the agent MUST re-run the check after any attempt to fix it. Never assume a "heal" without verification.
4. **Audit/Simulate Exemption**: In `/magic.analyze` (Ventilation) or `/magic.simulate` (Validation), caching is NOT permitted. These workflows must perform fresh, physical scans by definition to fulfill their audit purpose.

### C24 — Role-Switching Gates

At critical decision points, the agent MUST adopt a specific adversarial persona before finalizing output. This prevents confirmation bias and "glazed eye" failures where the agent that produced work also approves it.

| Workflow | Gate | Persona | Key Questions |
| :--- | :--- | :--- | :--- |
| `spec.md` | Before `Post-Update Review` | **Project Critic** | L1 purity? Invariant completeness? L2 compliance substantive? |
| `task.md` | Before `Plan Write-back` | **Planning Skeptic** | Optimism bias? Hidden dependencies? Cascade risk? |
| `run.md` | Before marking task `Done` | **Tester** | Spec boundary? Edge cases? Side effects? Regression risk? |
| `retrospective.md` | Before Signal calculation | **Independent Analyst** | Does Signal reflect spec quality, not just execution stats? |
| `analyze.md` | Before Advisory Report | **Auditor** | Severity correct? Systemic pattern behind findings? |
| `rule.md` | Before Impact Analysis | **Constitutional Reviewer** | Practical conflict with C1–C23 in running workflows? |

Switching is mandatory — it is not skipped in Trust Mode (C9). The persona switch takes one internal reasoning pass; it does not require user interaction.

### C25 — Generic Float Constraint

All numeric computation in the GoNN library MUST use the `utils.Float` type constraint (`float32 | float64`). No hardcoded float types are permitted in generic code. The `Float` constraint defined in `pkg/utils/float.go` is the single source of truth.

### C26 — Compile-Time Interface Verification

All concrete types implementing an interface MUST include a compile-time verification statement: `var _ InterfaceName[float32] = (*TypeName[float32])(nil)`. This prevents interface drift and catches missing method implementations at compile time rather than runtime.

### C27 — Dispatcher Pattern Convention

Enum-based dispatcher functions follow this structure: (a) `Type uint8` enum with `String()` method, (b) one implementation file per enum value, (c) switch-based dispatcher function. Adding a new variant requires all three steps. This pattern is used for activation functions and loss functions.

### C28 — Composition via Embedding

Type hierarchies use Go struct embedding for composition (not interface embedding or inheritance simulation). The pattern is: base unexported type → extended unexported type → exported specialized type. Each level adds capabilities without modifying the parent.

### C29 — Zero External Dependencies

The GoNN library MUST NOT introduce external dependencies. Only the Go standard library is permitted. Any proposal to add a third-party dependency requires explicit justification and approval.

The canonical source for permitted packages is the Go standard library tree at <https://cs.opensource.google/go>. When stdlib provides a primitive, no third-party alternative is permissible regardless of perceived ergonomic benefits. Test-only dependencies (e.g., `testify`) are not exempt — they leak into examples and onboarding cost.

### C30 — Test Coverage and Benchmarks

Every new Go package and exported function MUST be covered by tests:

1. **Coverage floor**: 80% line coverage per package, measured by `go test -cover`. New code below this floor blocks the merge.
2. **Table-driven tests**: required for any function with ≥3 distinct input scenarios. One subtest per case via `t.Run`.
3. **Race detection**: `go test -race ./...` must pass. Non-deterministic failures are bugs, not flakes.
4. **Benchmarks**: every hot path identified in `l1-performance-contract.md` (PERF-1) MUST have at least one `Benchmark*` function. Regressions > 10% vs. baseline block the merge.
5. **Fuzz tests**: input-validation boundaries (parsers, numeric guards) SHOULD have native Go fuzz tests (`Fuzz*`).

### C31 — Doc-comment Verbosity

Every exported identifier (type, function, method, variable, constant) MUST carry a Go doc comment. Per project rule §1.1, comments are English. The level of detail required scales with audience exposure:

1. **Public API** (`pkg/nn/`, `cmd/`): full doc comment — purpose, parameters, return semantics, error contract, ≥1 usage example for non-trivial functions.
2. **Internal exported** (other `pkg/`): purpose + parameter contract. Examples optional.
3. **Unexported**: short comment if the name is not self-evident; skip if the signature explains itself.

The first sentence of every doc comment MUST start with the identifier name (Go convention: `// Foo does ...`). Doc comments MUST explain **why** the function exists, not only **what** it does — readers understand mechanics from the code; they need intent from the comment.

### C32 — Error Informativeness

Per `l1-error-taxonomy.md` ERR-4, error messages MUST be specific and actionable:

1. **Forbidden phrases**: "something went wrong", "internal error", "unknown error", "invalid input" without specifying which input.
2. **Required content**: identify the offending value (or its name), the constraint violated, and where possible the remediation. Example: `Input(): size must be positive, got 0` (not `invalid size`).
3. **Wrapping**: use `fmt.Errorf("...: %w", err)` to preserve the chain. Never lose the root cause via `err.Error()` re-wrapping.
4. **Category sentinel**: every returned error wraps a category sentinel from `pkg/utils/errors.go` (`ErrUserConfig`, `ErrIntegrity`, etc.) so callers can route via `errors.Is`.
5. **Location hint**: at Error log level, include a short caller hint (`function or file:line`) when it improves diagnosability.

### C33 — AI-Meta Annotation

Exported identifiers (types, functions, methods, interfaces, error sentinels) MUST carry a trailing `AI-Meta:` block in their doc comment, on top of the verbosity required by C31. The block uses a closed vocabulary and gives AI assistants and human readers constant-time orientation per symbol. Full grammar and field semantics live in `l2-ai-doc-metadata.md`.

1. **Format**: trailing labeled list, separated from preceding prose by one blank doc-comment line. Each entry is `  - <Field>: <single-line value>`. Block ends the doc comment.
2. **Closed vocabulary** (no other field names): `Purpose`, `Usage`, `Lifecycle`, `Concurrency`, `Errors`, `Related`, `Constraints`, `Implementations`, `Stability`.
3. **Closed enums**: `Stability` ∈ `Stable | Experimental | Deprecated | Internal`. `Concurrency` ∈ `Safe | ReadSafe | SingleGoroutine | NotSafe` (optional `; <clarifier>` suffix).
4. **Tier-gated obligations** (mirrors C31 audience tiers): public API (`pkg/nn/`, `cmd/`) — full block; other exported (`pkg/*`) — `Purpose` and condition-driven fields; unexported — block omitted.
5. **Process-artifact firewall**: block content MUST NOT cite SDD artifacts. No `.design/...` paths, no spec filenames, no `INV-N` invariant numbers, no `C-rule` numbers. Each line is self-contained natural language. Sentinel error names, state names, and other identifiers exported by the library are permitted — they are part of the public surface.
6. **Hard cap**: ≤ 12 lines including the `AI-Meta:` label. Going over signals over-documentation; trim or split the symbol.
7. **Verification**: until the dedicated linter (`cmd/lint-aimeta`) lands, compliance is verified via `go doc -all` rendering check and reviewer checklist. The linter, when delivered, is invoked from per-package `TestAIMetaCompliance` so the convention is enforced by `go test ./...`.

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-04-21 | Initial constitution |
| 1.1.0 | 2026-04-21 | Added C25-C29 project conventions from codebase analysis |
| 1.2.0 | 2026-04-27 | Added C30 (Test Coverage), C31 (Doc-comment Verbosity), C32 (Error Informativeness) from TODO.md ideation. Enhanced C29 with stdlib canonical reference. |
| 1.3.0 | 2026-05-07 | Added C33 (AI-Meta Annotation) from TODO #27. Closed vocabulary, tier-gated obligations, process-artifact firewall. Full grammar in `l2-ai-doc-metadata.md`. |
