# Checkpointing Implementation

**Version:** 0.1.0
**Status:** Draft
**Layer:** implementation
**Implements:** l1-checkpointing.md

## Overview

Concrete Go realization of [l1-checkpointing.md](l1-checkpointing.md): the `pkg/checkpoint/` package, atomic-write helpers (temp + fsync + rename), `compress/gzip` cold-tier compression, retention sweeper goroutine, and the schema extension over `l2-persistence-impl.md` that adds RNG-state and iteration-counter fields.

## Related Specifications

- [l1-checkpointing.md](l1-checkpointing.md) — Parent — invariants CHK-1..CHK-4 + storage layout
- [l2-persistence-impl.md](l2-persistence-impl.md) — Reused JSON schema for weights + config
- [l2-control-impl.md](l2-control-impl.md) — Pause event triggers `flush()` before transitioning to Paused
- [l2-init-impl.md](l2-init-impl.md) — RNG state is `*rand.PCG.MarshalBinary()`

## 1. Motivation

L1 fixes triggers, retention policy, and the atomic-write contract. This spec decides Go specifics — package boundary, file naming and locking, the exact extension of the persistence schema for resume-from-snapshot, and how the retention sweep is scheduled without blocking training.

## 2. Constraints & Assumptions

- Stdlib only: `os`, `path/filepath`, `compress/gzip`, `encoding/json`, `time`.
- Snapshots are written from the **training goroutine** — no separate writer pool. Disk I/O is part of the iteration budget; `WithSnapshotInterval` defaults are tuned so cost is amortized.
- Retention sweep runs at most once per minute on a separate goroutine; missed cycles are merged on the next tick.
- Atomic write: temp file in same directory → `Sync()` → `Rename` to target. Cross-FS rename is not supported.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| CHK-1 (Atomic write) | `os.CreateTemp(dir, ".snap-*.json.tmp")` → write → `Sync` → `Rename` → atomic on POSIX/NTFS |
| CHK-2 (Resumable contents) | Snapshot embeds `iter`, `rng_state` (PCG MarshalBinary), `min_loss_state`, plus full weights+config from l2-persistence-impl |
| CHK-3 (Retention) | Sweeper enumerates `snap-*.json{,.gz}` by mtime; keeps last N hot, gzips next M, deletes rest |
| CHK-4 (Schema versioning) | Top-level `schema_version: "1.0"`; `migrate(from, to, raw)` registry indexed by version pair |

## 5. Detailed Design

### 5.1 Package Layout

```text
pkg/checkpoint/
├── snapshot.go    // Snapshot struct + JSON marshal
├── writer.go      // atomic write + gzip helpers
├── reader.go      // detect plain vs gzip; load and dispatch migrate()
├── retention.go   // sweep goroutine + policy
└── checkpoint_test.go
```

### 5.2 Snapshot Struct

```go
// [REFERENCE] In pkg/checkpoint/snapshot.go.
type Snapshot[T utils.Float] struct {
    SchemaVersion string                  `json:"schema_version"`
    Iter          uint64                  `json:"iter"`
    Loss          T                       `json:"loss"`
    MinLossState  MinLossState[T]         `json:"min_loss_state"`
    RNGState      []byte                  `json:"rng_state"` // base64 in JSON
    Config        persistence.Config[T]   `json:"config"`
    Weights       persistence.Weights[T]  `json:"weights"`
    Timestamp     int64                   `json:"timestamp"` // unix
}
```

### 5.3 Atomic Write (pseudo-code)

```text
WriteSnapshot(dir, snap):
    name = fmt.Sprintf("snap-%d-%d.json", snap.Iter, snap.Timestamp)
    tmp, err = os.CreateTemp(dir, ".snap-*.json.tmp")
    if err != nil: return wrap(ErrIO, err)
    defer os.Remove(tmp.Name())  // best-effort cleanup if rename fails
    enc = json.NewEncoder(tmp)
    enc.Encode(snap)
    tmp.Sync()
    tmp.Close()
    return os.Rename(tmp.Name(), filepath.Join(dir, name))
```

### 5.4 Retention Sweep

```text
sweep(dir, hotN=3, coldM=10):
    files = list snap-*.json (sorted by mtime desc)
    keep_hot = files[:hotN]
    promote_cold = files[hotN:hotN+coldM]
    for f in promote_cold:
        if not gzipped: gzip(f) → f + ".gz"; remove f
    delete = files[hotN+coldM:]
    for f in delete: os.Remove(f)
```

### 5.5 Open Questions

- <!-- TBD: snapshot every N iterations vs every M seconds — both. Take whichever fires first; reset both clocks on snapshot. -->
- <!-- TBD: what happens if disk is full? CHK-1 says best-effort + log; need to wire into l2-logging-strategy with Warn level + ErrIO. -->
- <!-- TBD: snapshot during Stopping — write final state or skip? Lean toward write (last chance). -->

## 6. Implementation Notes

1. New package `pkg/checkpoint/`; depends on `pkg/persistence` for shared schema.
2. RNG marshaling: `pcg.MarshalBinary()` returns 16 bytes; base64-encoded inside JSON. `UnmarshalBinary` reverses.
3. Retention sweeper: `time.NewTicker(1 * time.Minute)`; receive on training-loop done channel to terminate cleanly.
4. Tests require a temp directory; use `t.TempDir()` for hermetic isolation.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[CKPT]` | `pkg/checkpoint/` (new) | Snapshot writer/reader/retention |
| `[PERS]` | `pkg/persistence/` (new) | Reused schema (l2-persistence-impl) |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-28 | Initial Draft — concrete Go realization of l1-checkpointing RFC. |
