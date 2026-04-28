# Data Streaming & Bounded Memory

**Version:** 0.1.0
**Status:** RFC
**Layer:** concept

## Overview

Defines a streaming `Dataset` abstraction for GoNN so training scales to data larger than RAM. The
network consumes mini-batches via a channel- or iterator-based source rather than receiving the full
dataset upfront. Establishes memory bounds, batching contract, and shuffle semantics.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent
- [l1-performance-contract.md](l1-performance-contract.md) — Streaming feeds the worker pool
- [l1-training-control.md](l1-training-control.md) — Pause halts dataset advance at safe points

## 1. Motivation

Today `Train(input, target)` accepts only one sample. There is no facility for:

- Iterating a dataset that doesn't fit in RAM (image archives, log files).
- Mini-batch training (the standard practice in modern NN).
- Shuffling, augmentation, or lazy decoding of samples.
- Backpressure from the training loop to the data source.

A streaming abstraction unifies these without forcing the library to know about file formats.

## 2. Constraints & Assumptions

- Library defines an interface; format-specific adapters (CSV, image folders, MNIST) are user code or
  thin helpers in `pkg/dataset/`.
- Memory bound is configurable via `WithBatchSize(n)` and `WithPrefetch(k)` — the library never
  buffers more than `n × (k+1)` samples.
- Channel-based streaming preferred — natural backpressure, integrates with goroutine pools.
- Shuffle is **opt-in** and seeded for reproducibility.

## 3. Core Invariants

- **DAT-1**: A `Dataset[T]` is a pull source: caller advances iteration; source decides when data is
  loaded. Library never seeks or restarts a dataset.
- **DAT-2**: A `Batch[T]` is a homogeneous group of `(input, target)` pairs all sharing the same
  shape. Mixing shapes within a batch is undefined.
- **DAT-3**: Memory residency is **bounded by configuration**. The library holds at most
  `(prefetch + 1) × batchSize` samples at once.
- **DAT-4**: Dataset errors are **fatal to the current epoch**, not the network. The training loop
  surfaces the error with `ErrInputData` (per `l1-error-taxonomy.md`); the network remains queryable.

## 5. Detailed Design

### 5.1 Interface Sketch

```text
type Dataset[T utils.Float] interface {
    Next(ctx) (Batch[T], error)  // io.EOF signals epoch end
    Reset(ctx) error              // optional — for streamable epoch restart
    Len() (int, bool)             // returns -1, false if unknown (true streaming)
}

type Batch[T utils.Float] struct {
    Inputs  [][]T  // [batchSize][inputSize]
    Targets [][]T  // [batchSize][outputSize]
}
```

### 5.2 Open Questions

- <!-- TBD: how does Train(...) accept a Dataset alongside legacy single-sample (input, target)? Overload via builder option? -->
- <!-- TBD: parallel decoding — pre-fetcher goroutine pool? -->
- <!-- TBD: integration with `l1-checkpointing.md` — checkpoint must record epoch position for resume -->

## 6. Implementation Notes

1. Phase 1: define the `Dataset` / `Batch` types in `pkg/dataset/`.
2. Phase 2: provide `NewSliceDataset(samples)` adapter wrapping the existing single-sample API.
3. Phase 3: provide `NewCSVDataset(path)` for the canonical example dataset format.
4. Phase 4: prefetch goroutine and `WithPrefetch(k)` option.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[DATASET]` | `pkg/dataset/` (new) | Streaming abstraction home |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #12. |
| 0.1.0 | 2026-04-28 | Status promoted Draft → RFC. Dataset/Batch interface and 4 invariants ready for review. |
