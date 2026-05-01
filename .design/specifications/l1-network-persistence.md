# Network Persistence

**Version:** 1.0.0
**Status:** Stable
**Layer:** concept

## Overview

Defines the on-disk representation of a GoNN network — split into a **config** document
(architecture and hyperparameters, human-editable) and a **weights** document (trained values,
machine-only). Establishes the JSON schema, version field semantics, and round-trip integrity
guarantees that downstream specs (`l1-checkpointing.md`, `l2-cli-client.md`, the E09 example) depend on.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent
- [l1-checkpointing.md](l1-checkpointing.md) — Snapshots are persistence + RNG state
- [l2-nn-facade.md](l2-nn-facade.md) — `Config[T]` (§5.5) is the in-memory shape this spec serializes
- [l2-cli-client.md](l2-cli-client.md) — Consumes config/weights files

## 1. Motivation

A trained network is valuable artifact. Without a stable on-disk format users cannot:

- Share trained models (no canonical file to email, attach to issue, or commit to a git LFS).
- Resume training after machine reboot (gated by `l1-checkpointing.md`).
- Run a model in production from a CLI (gated by `l2-cli-client.md`).
- A/B-compare two trained networks (no diffable source).

Splitting **config** from **weights** is deliberate: configs are small, human-readable, diffable and
version-controllable; weights are large, opaque, and version-controlled separately or via LFS.

## 2. Constraints & Assumptions

- JSON for both documents — chosen for tooling ubiquity (jq, IDE support) and self-describing layout.
- Stdlib `encoding/json` only (per `C29`).
- Generic types serialize via Go's standard JSON — `T utils.Float` becomes a plain JSON number.
- Forward compatibility: an older library reading a newer file with extra fields ignores them
  (`json.Decoder` default behavior). Reverse is **not** guaranteed and is governed by §3 below.

## 3. Core Invariants

- **PERS-1**: Every persisted document carries a `schema_version` field (semver). Major version mismatch
  on load → fatal error (`ErrUserConfig`); minor / patch mismatch → warning + best-effort load.
- **PERS-2**: Config document is **deterministic** — re-serializing a loaded config produces a
  byte-identical file (sorted keys, stable map iteration). This makes configs diffable.
- **PERS-3**: Weights document references its companion config via an embedded `config_hash` field
  (SHA-256 of the canonical config bytes). Loading weights against a non-matching config is a
  `ErrIntegrity` error.
- **PERS-4**: Round-trip integrity for `float32`: `Read(Write(network))` produces a network that
  scores within **1 ULP** of the original on the verification dataset. For `float64`: bit-identical.

## 5. Detailed Design

### 5.1 File Layout

```plaintext
{model-name}/
├── config.json   # architecture + hyperparameters, ≈ 1 KiB
├── weights.json  # trained values, scales with topology
└── meta.json     # optional — training timestamps, git rev, dataset hash
```

CLI / library accept either the directory path (auto-discovers files) or the individual file paths.

### 5.2 Config Schema (proposed)

```json
{
  "schema_version": "1.0.0",
  "lib_version": "0.x.y",
  "precision": "float32",
  "input_size": 2,
  "hidden_layers": [
    {"size": 4, "activation": "Sigmoid", "bias": true}
  ],
  "output": {"size": 1, "activation": "Sigmoid", "bias": true},
  "training": {
    "learning_rate": 0.3,
    "loss": "MSE",
    "loss_limit": 1e-4,
    "max_iterations": 10000,
    "weight_init": "xavier",
    "rng_seed": 42
  }
}
```

Field naming: snake_case in JSON (idiomatic for cross-language), camelCase in Go bindings via struct tags.

### 5.3 Weights Schema (proposed)

```json
{
  "schema_version": "1.0.0",
  "config_hash": "sha256:...",
  "layers": [
    {
      "name": "hidden_0",
      "weights": [[...], [...]],
      "biases": [...]
    }
  ]
}
```

Weight matrices serialize as JSON arrays of arrays; for very large networks a `weights.bin` companion
(little-endian raw floats) MAY be referenced by future v2 schema — out of scope for v1.

### 5.4 Open Questions

- <!-- TBD: include optimizer state in weights.json (when optimizers land) or separate file? -->
- <!-- TBD: compression — gzip on disk for large weights, or expect users to wrap manually? -->
- <!-- TBD: what survives migration across MAJOR schema bumps — define migration function signature -->

## 6. Implementation Notes

1. Stage 1: define `pkg/persistence/config.go` and `pkg/persistence/weights.go` with `Read`/`Write`
   functions.
2. Stage 2: integrate `WriteConfig`/`ReadConfig` and `WriteWeights`/`ReadWeights` into `pkg/nn/nn.go`
   (per `l2-nn-facade.md` Implementation Notes).
3. Stage 3: `gonn save` and `gonn load` CLI commands (per `l2-cli-client.md`).

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[PERSIST]` | `pkg/persistence/` (new) | Reader/writer home |
| `[REF-OLD]` | `.references/gonn_old/pkg/nn/write.go` | Historical writer |
| `[REF-RU]` | `.references/rustunumic/cells.json` | Sample weight serialization shape |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft — closes references from l1-checkpointing, l2-cli-client, l2-usage-examples E09. |
| 0.1.0 | 2026-04-27 | Status promoted Draft → RFC. Config and weights schemas defined, integrity invariants (PERS-1..PERS-4) ready for review. |
| 1.0.0 | 2026-05-01 | [Batch-Stabilize] RFC → Stable. MVC satisfied: Overview + Core Invariants PERS-1..4. C9 Trust Mode auto-promotion. |
