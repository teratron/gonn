# Persistence Implementation

**Version:** 0.1.0
**Status:** Draft
**Layer:** implementation
**Implements:** l1-network-persistence.md

## Overview

Concrete realization of [l1-network-persistence.md](l1-network-persistence.md) in Go: the `pkg/persistence`
package with `ReadConfig` / `WriteConfig` / `ReadWeights` / `WriteWeights` functions, the canonical
JSON marshal/unmarshal logic, atomic file write protocol, and `config_hash` computation.

## Related Specifications

- [l1-network-persistence.md](l1-network-persistence.md) — Parent — invariants PERS-1..PERS-4
- [l1-checkpointing.md](l1-checkpointing.md) — Reuses these readers/writers for snapshot files
- [l2-cli-client.md](l2-cli-client.md) — Consumer of these functions

## 1. Motivation

L1 fixes the wire format and integrity rules. This spec pins the Go structures, error paths, and
filesystem mechanics — the parts that cannot be portable but must be uniform inside the project.

## 2. Constraints & Assumptions

- Stdlib only: `encoding/json`, `crypto/sha256`, `os`, `io`.
- Atomic write: temp file in same directory + `os.Rename` after `Sync()` (POSIX guarantee).
- Reader is forgiving for forward compat: unknown JSON fields are silently ignored
  (`json.Decoder` default).

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| PERS-1 (schema_version) | Decoded `schema_version` parsed via stdlib semver-lite; major mismatch returns `ErrUserConfig`-wrapped error. |
| PERS-2 (Determinism) | Marshal uses sorted keys via custom encoder pass; map iteration replaced with sorted-slice projection. |
| PERS-3 (config_hash) | SHA-256 over canonical config bytes; embedded in `weights.json`; verified on load. |
| PERS-4 (Round-trip ε) | Float round-trip uses `strconv.FormatFloat(v, 'g', -1, 32/64)` to preserve ULP-1 precision for f32 / bit-identical for f64. |

## 5. Detailed Design

### 5.1 Public Surface

```go
// [REFERENCE] Functions in pkg/persistence/.
func WriteConfig[T utils.Float](path string, cfg Config[T]) error
func ReadConfig[T utils.Float](path string) (Config[T], error)
func WriteWeights[T utils.Float](path string, cfg Config[T], w WeightsDoc[T]) error
func ReadWeights[T utils.Float](configPath, weightsPath string) (Config[T], WeightsDoc[T], error)
```

`Config[T]` mirrors the in-memory `pkg/nn.Config` from `l2-nn-facade.md` §5.5 with JSON struct tags.

### 5.2 Atomic Write Pseudo-code

```text
WriteConfig(path, cfg):
    tmp = path + ".tmp." + randomSuffix()
    f, err = os.Create(tmp); if err: return wrap(ErrIO, err)
    defer cleanup(tmp)
    enc = json.NewEncoder(f); enc.SetIndent("", "  ")
    if err = enc.Encode(canonicalize(cfg)): return wrap(ErrIO, err)
    if err = f.Sync(): return wrap(ErrIO, err)
    if err = f.Close(): return wrap(ErrIO, err)
    if err = os.Rename(tmp, path): return wrap(ErrIO, err)
    return nil
```

### 5.3 Canonicalization

`canonicalize(cfg)` produces a stable byte representation by:

1. Sorting all map keys alphabetically before serialization.
2. Emitting numeric values via `strconv.FormatFloat` with `-1` precision (round-trip exact).
3. Using `json.RawMessage` for embedded sub-documents to avoid double-encoding drift.

The canonical form is what feeds `sha256.Sum256(...)` for `config_hash`.

### 5.4 Open Questions

- <!-- TBD: BOM/encoding policy — UTF-8 enforced; reject any BOM? -->
- <!-- TBD: large weights — when do we add a binary fallback (`weights.bin` + JSON metadata)? size threshold TBD -->

## 6. Implementation Notes

1. New package `pkg/persistence/` — keeps `pkg/nn/` thin.
2. Hash computation uses `sha256.New()` streamed over the canonical bytes — no full materialization.
3. Errors wrap category sentinels per `l1-error-taxonomy.md` (ERR-3): `ErrUserConfig` for schema
   issues, `ErrIntegrity` for hash mismatch, `ErrIO` for filesystem.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[PERSIST]` | `pkg/persistence/` | Implementation home |
| `[ERRORS]` | `pkg/utils/errors.go` | Sentinel categories used by readers/writers |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-28 | Initial Draft — concrete Go realization of l1-network-persistence RFC. |
