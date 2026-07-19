# GoNN Developer Skills

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-neural-network-architecture.md

## Overview

Defines the structure and content of GoNN developer skills — AI-agent instruction sets
that enable tools (Claude, Copilot, Gemini, Cursor, etc.) to generate high-performance
GoNN code following library best practices. The skills live in a `skills/` directory at
the project root and follow the standard `SKILL.md` format used by AI agent ecosystems.

This is a **tooling** spec: it does not add runtime code to the library but creates
developer-experience artifacts that amplify adoption and correct usage.

## Related Specifications

- [l2-nn-facade.md](l2-nn-facade.md) — The API surface skills teach users to use
- [l2-deep-builder.md](l2-deep-builder.md) — Deep builder ergonomics skills should cover
- [l2-usage-examples.md](l2-usage-examples.md) — Canonical examples that skills reference
- [l2-ai-doc-metadata.md](l2-ai-doc-metadata.md) — AI-Meta annotations skills should generate
- [l1-lr-scheduling.md](l1-lr-scheduling.md) — Scheduler patterns skills should teach

## 1. Motivation

GoNN is a Go library with dual API styles, type generics, multiple activation/loss functions,
optimizer strategies, and layer composition patterns. Without guided instruction, AI assistants:

- Generate invalid API usage (mixing Builder and Options styles).
- Forget `Compile()` / `MustCompile()` finalization.
- Use raw `float64` instead of the `utils.Float` constraint.
- Miss AI-Meta annotations on exported identifiers.
- Hardcode layer sizes instead of using bulk constructors for deep networks.

A curated skill set eliminates these failures by embedding GoNN's conventions directly into
the AI's instruction context.

## 2. Constraints & Assumptions

- Skills are **read-only resources** — they instruct AI agents but contain no executable code.
- Skills MUST NOT reference SDD artifacts (`.design/`, `.magic/`) — they are user-facing.
  This mirrors the process-artifact firewall from `l2-ai-doc-metadata.md` C33 §5.
- Skills follow the `SKILL.md` YAML frontmatter + markdown body format.
- Skills are versioned alongside the library — skill content matches the library's public API.
- The `skills/` directory is at project root, NOT inside `.agents/` (which is the internal
  SDD agent config). User-facing skills are separate from internal agent workflows.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| INV-1 (Generic Float) | Skills teach `utils.Float` constraint usage; all examples use `T`. |
| INV-6 (Interface segregation) | Skills reference only public `pkg/nn/` API. |

## 5. Detailed Design

### 5.1 Directory Structure

```plaintext
skills/
└── gonn/
    ├── SKILL.md              # Main skill — GoNN code generation
    ├── examples/
    │   ├── builder-xor.md    # Builder Style A example
    │   ├── options-mnist.md  # Options Style B example
    │   ├── deep-network.md   # 100-layer network construction
    │   └── custom-training.md # Optimizer + scheduler + callbacks
    └── resources/
        ├── api-reference.md  # Condensed public API surface
        ├── activation-guide.md  # When to use which activation
        ├── loss-guide.md     # When to use which loss function
        └── conventions.md    # GoNN-specific coding conventions
```

### 5.2 Core Skill Content (SKILL.md)

The main `SKILL.md` must cover:

1. **API Style Selection**: When to use Builder (tutorials, simple) vs Options (advanced, reuse).
2. **Construction Lifecycle**: `New → Configure → Compile → Train/Query`.
3. **Type Parameterization**: Always use `[float32]` or `[float64]` — never raw floats.
4. **Bulk Constructors**: `Repeat`, `Pattern`, `WithHiddenLayers` for deep networks.
5. **Activation ↔ Loss Matching**: Sigmoid→BCE, Softmax→CE, Linear→MSE.
6. **Optimizer Selection**: SGD for simple, Adam for deep, with LR scheduler pairing.
7. **AI-Meta Generation**: How to write C33-compliant doc-comment annotations.
8. **Error Handling**: All GoNN errors wrap category sentinels — use `errors.Is()`.
9. **Testing Patterns**: Table-driven tests, benchmark requirements, race detection.
10. **Common Mistakes**: Missing `Compile()`, mixing styles, wrong loss/activation pairs.

### 5.3 SKILL.md Frontmatter

```yaml
---
name: gonn
description: >
  Generate high-performance GoNN neural network code following library best practices.
  Covers Builder and Options API styles, deep network construction, optimizer/scheduler
  configuration, and AI-Meta doc-comment annotations.
---
```

### 5.4 Example Reference Files

Each example file is a self-contained markdown document with:

- **Goal**: What the example builds (e.g., "XOR classifier with Builder API").
- **Code**: Complete, runnable Go code with `// Output:` annotations.
- **Explanation**: Step-by-step walkthrough of API choices.
- **Anti-patterns**: What NOT to do and why.

### 5.5 API Reference Resource

A condensed reference covering:

- All public types (`NN[T]`, `Config[T]`, `HiddenLayerSpec[T]`, `Option[T]`).
- All builder methods with signatures and brief descriptions.
- All option constructors with signatures.
- All higher-order options and presets.
- Default values for all configuration parameters.

This file is optimized for AI context window efficiency — no prose, just structured
signatures and descriptions.

### 5.6 Conventions Resource

GoNN-specific coding conventions that AI agents must follow:

- `utils.Float` constraint (C25).
- Compile-time interface verification (C26).
- Dispatcher pattern for activations/losses (C27).
- Composition via embedding (C28).
- Zero external dependencies (C29).
- Test coverage and benchmarks (C30).
- Doc-comment verbosity (C31).
- Error informativeness (C32).
- AI-Meta annotation (C33).

**Note**: This resource translates RULES.md conventions into AI-friendly instructions
without referencing RULES.md itself (process-artifact firewall).

## 6. Implementation Notes

1. Create `skills/gonn/SKILL.md` first — this is the minimum viable skill.
2. Add `examples/` files from existing `examples/` directory — translate Go files to
   annotated markdown.
3. Add `resources/` files by extracting public API surface from `pkg/nn/` source.
4. Reference the skill in `README.md` — add a "Use with AI Tools" section.
5. Skill content MUST be updated when the public API changes (facade amendments,
   new constructors, new activation/loss functions).

## 7. Drawbacks & Alternatives

- **Alternative (In-code doc-comments only)**: Rely entirely on `go doc` output and AI-Meta.
  This works for code-aware agents but fails for agents that need higher-level guidance
  (when to use Builder vs Options, architecture recommendations).
- **Alternative (README-only)**: Expand README.md with all skill content. This bloats the
  README and is not structured for AI consumption.
- **Drawback (Maintenance burden)**: Skills must track API changes. Mitigation: skill
  content references code examples that are already maintained; API reference is generated
  from source when possible.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[SKILL]` | `skills/gonn/SKILL.md` | Main skill file — GoNN code generation instructions |
| `[FACADE]` | `.design/main/specifications/l2-nn-facade.md` | API surface that the skill teaches |
| `[EXAMPLES]` | `examples/` | Canonical Go examples that skill references mirror |
| `[README]` | `README.md` | Library entry point — will reference the skill |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-05-08 | Initial — GoNN developer skills structure and content plan. Trust Mode Stable. |
