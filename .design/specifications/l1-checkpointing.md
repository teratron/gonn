# Checkpointing & Snapshot Lifecycle

**Version:** 0.1.0
**Status:** RFC
**Layer:** concept

## Overview

Defines automatic snapshots of training state for crash recovery, resume-from-pause, and rollback.
Snapshots capture weights + optimizer state + RNG seed + iteration counter, persisted as JSON.
Old snapshots are compressed and rotated under a retention policy to bound disk usage.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent invariants
- [l1-network-persistence.md](l1-network-persistence.md) — Snapshot reuses the config + weights schema
- [l1-training-semantics.md](l1-training-semantics.md) — Min-loss snapshot interacts with weight rollback (TRN-3)
- [l1-training-control.md](l1-training-control.md) — Pause triggers snapshot; resume reads latest

## 1. Motivation

Training a non-trivial network can take hours. Power loss, OOM kill, manual Ctrl-C, or simply a
broken assumption shouldn't waste that compute. Snapshots make training **cheap to interrupt and
resume**, which in turn makes long runs viable on consumer hardware (laptop on battery, cloud spot
instances).

## 2. Constraints & Assumptions

- JSON format for human-readability and tool compatibility (jq, scripts).
- Snapshot files named `snap-{iteration}-{unix-ts}.json`; latest also symlinked / copied to `latest.json`.
- Configurable interval (every N iterations or every M seconds — whichever first).
- Compression applied to old snapshots after they fall outside the "hot retention window".
- All file I/O is best-effort: snapshot failure must not crash training (log + continue).

## 3. Core Invariants

- **CHK-1**: Every snapshot is **atomic** — written to a temp file, fsynced, then renamed. Readers
  never observe a partial snapshot.
- **CHK-2**: Snapshot contents are **complete enough to resume bit-identically**: weights, optimizer
  state (when optimizers land), RNG seed, current iteration counter, current min-loss tracking state.
- **CHK-3**: **Retention policy** is layered: keep last N hot (uncompressed), keep last M cold
  (compressed), delete the rest. Defaults: N=3, M=10.
- **CHK-4**: **Backwards compatibility** — every snapshot embeds a schema version. Older snapshots
  may be read by a `migrate` function until support is explicitly dropped (1 major version of grace).

## 5. Detailed Design

### 5.1 Snapshot Trigger Sources

| Source | Trigger |
| :--- | :--- |
| Interval | Every N iterations OR every M seconds since last snapshot |
| Pause event | Force-flush current state before transitioning to Paused (per `l1-training-control` CTRL-2) |
| Min-loss improvement | Optional — snapshot when a new minimum loss is reached |
| Explicit `Snapshot()` call | User-driven from API |

### 5.2 Storage Layout

```plaintext
{checkpoint_dir}/
├── latest.json              # Current — atomic rename target
├── snap-100-1714214400.json # Hot retention
├── snap-90-1714213500.json
├── snap-80-1714212600.json
├── snap-70-1714211700.json.gz  # Cold retention (gzipped)
├── snap-60-1714210800.json.gz
└── ...
```

### 5.3 Open Questions

- <!-- TBD: schema version field name and bumping policy -->
- <!-- TBD: integration with l1-network-persistence (when drafted) — share format? -->
- <!-- TBD: distributed training? probably out of scope for v1 -->

## 6. Implementation Notes

1. Reuse the JSON schema from [l1-network-persistence.md](l1-network-persistence.md) §5.2/§5.3; snapshot adds RNG state and iteration counter as additional top-level fields.
2. Atomic write: use `os.Rename` after `Sync()` — POSIX guarantee.
3. Compression via stdlib `compress/gzip` (per `C29 — Zero External Dependencies`).

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[REF-OLD]` | `.references/gonn_old/pkg/nn/write.go` | Historical config/weights writers — pattern to evolve |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #3. |
| 0.1.0 | 2026-04-28 | Cross-refs to network-persistence and training-semantics added. Status promoted Draft → RFC. |
