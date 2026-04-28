# Data Streaming Implementation

**Version:** 0.1.0
**Status:** Draft
**Layer:** implementation
**Implements:** l1-data-streaming.md

## Overview

Concrete Go realization of [l1-data-streaming.md](l1-data-streaming.md): the `pkg/dataset/` package, the `Dataset[T]` / `Batch[T]` types, the `NewSliceDataset` and `NewCSVDataset` adapters, the prefetch goroutine driven by `WithPrefetch(k)`, and the integration point in `Train(ctx, ds)` that replaces single-sample call sites.

## Related Specifications

- [l1-data-streaming.md](l1-data-streaming.md) — Parent — invariants DAT-1..DAT-4 + interface sketch
- [l2-training-loop.md](l2-training-loop.md) — Consumer of `Dataset[T].Next()`
- [l2-control-impl.md](l2-control-impl.md) — `Pause` halts dataset advance at the safe point
- [l2-errors-impl.md](l2-errors-impl.md) — Dataset errors wrap `ErrInputData` per DAT-4

## 1. Motivation

L1 fixes the abstraction and memory bound. This spec decides Go specifics — interface placement, channel-vs-callback for prefetch, error mapping into the taxonomy, and the migration plan for the existing single-sample `Train(input, target)` call sites.

## 2. Constraints & Assumptions

- Stdlib only: `context`, `encoding/csv`, `os`, `bufio`, `io`.
- Channel-based prefetch with goroutine cap = 1 prefetcher per dataset (DAT-3 memory bound respected).
- `Reset(ctx)` is optional — sources that can't seek (true streams) return `ErrUnsupported`.
- `Len()` returns `(-1, false)` for unknown size (true streaming) per DAT-1.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| DAT-1 (Pull source) | `Next(ctx) (Batch[T], error)`; library never calls `Reset` unless explicitly requested |
| DAT-2 (Homogeneous batch) | `Batch.Inputs` and `Batch.Targets` are `[][]T` of equal outer length; inner shape validated at first call |
| DAT-3 (Bounded memory) | Prefetcher has a `chan Batch[T]` of capacity `prefetch+1`; producer blocks on full channel (back-pressure) |
| DAT-4 (Errors fatal to epoch) | All non-`io.EOF` errors wrapped with `ErrInputData` and returned from `Train()`; network state untouched |

## 5. Detailed Design

### 5.1 Package Layout

```text
pkg/dataset/
├── dataset.go      // Dataset[T], Batch[T] interface + types
├── slice.go        // NewSliceDataset (in-memory wrapper)
├── csv.go          // NewCSVDataset (streaming CSV reader)
├── prefetch.go     // prefetch[T] decorator
└── dataset_test.go
```

### 5.2 Interface

```go
// [REFERENCE] In pkg/dataset/dataset.go.
type Dataset[T utils.Float] interface {
    Next(ctx context.Context) (Batch[T], error)
    Reset(ctx context.Context) error
    Len() (int, bool)
}

type Batch[T utils.Float] struct {
    Inputs  [][]T
    Targets [][]T
}
```

### 5.3 Prefetch Decorator

```go
func Prefetch[T utils.Float](inner Dataset[T], k int) Dataset[T] {
    p := &prefetch[T]{inner: inner, ch: make(chan result[T], k+1)}
    go p.run()  // single producer goroutine
    return p
}

type result[T utils.Float] struct { batch Batch[T]; err error }

func (p *prefetch[T]) Next(ctx context.Context) (Batch[T], error) {
    select {
        case r := <-p.ch: return r.batch, r.err
        case <-ctx.Done(): return Batch[T]{}, ctx.Err()
    }
}
```

### 5.4 Slice Adapter (Migration Bridge)

```go
func NewSliceDataset[T utils.Float](inputs, targets [][]T, batchSize int) Dataset[T] {
    return &sliceDataset[T]{inputs: inputs, targets: targets, batch: batchSize}
}
```

Existing call sites that pass `(input, target)` slices migrate to `NewSliceDataset(slices, 1)` for batch-size-1 backwards compatibility. The single-sample `Train()` overload is removed in v2.0 of `l2-nn-facade`.

### 5.5 Open Questions

- <!-- TBD: shuffle implementation — index slice + Fisher-Yates? Use the same RNG as l2-init-impl for reproducibility. -->
- <!-- TBD: image/byte datasets — separate package `pkg/dataset/image` or pluggable decoder hook? -->
- <!-- TBD: integration with checkpointing — record dataset cursor in snapshot for resume; needs Dataset.Cursor() optional method. -->

## 6. Implementation Notes

1. New package `pkg/dataset/`.
2. CSV adapter uses `bufio.Scanner` with a configurable max-line-length (default 1 MiB) — per `C29` no third-party CSV lib.
3. Prefetcher goroutine terminates on `ctx.Done()` or inner `io.EOF`; closes the channel cleanly.
4. Tests use `iotest.OneByteReader` and `iotest.ErrReader` to verify error propagation through DAT-4.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[DATASET]` | `pkg/dataset/` (new) | Streaming abstraction home |
| `[ERR]` | `pkg/utils/errors.go` | `ErrInputData` sentinel for DAT-4 wrapping |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-28 | Initial Draft — concrete Go realization of l1-data-streaming RFC. |
