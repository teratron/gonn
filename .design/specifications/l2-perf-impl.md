# Performance Implementation

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-performance-contract.md

## Overview

Concrete Go realization of [l1-performance-contract.md](l1-performance-contract.md): the benchmark file layout, the `sync.Pool` allocation conventions, preallocation hooks inside `Compile()`, the worker-pool sizing wired to `runtime.NumCPU()`, and the opt-in `net/http/pprof` hook surface.

## Related Specifications

- [l1-performance-contract.md](l1-performance-contract.md) — Parent — invariants PERF-1..PERF-5 + optimization layers
- [l2-backend-cpu.md](l2-backend-cpu.md) — Layer 5 (compute backend) lives there; this spec covers Layers 1–4
- [l2-training-loop.md](l2-training-loop.md) — Allocation hot path being optimized
- [RULES.md §C30](../RULES.md) — 80% coverage + benchmarks rule this spec operationalizes

## 1. Motivation

L1 fixes the contract. This spec decides Go specifics — file naming for benchmarks, what `sync.Pool` types are needed, preallocation strategy for transient buffers, and CI-side regression detection (which tool, which threshold, which command).

## 2. Constraints & Assumptions

- Stdlib only: `testing`, `sync`, `runtime`, `net/http/pprof` (opt-in).
- Benchmarks live alongside code: `pkg/network/network_bench_test.go`, etc.
- CI uses `benchstat` (separate `golang.org/x/perf` tool) to compare PR vs base. Threshold: `>10%` regression on any tracked benchmark fails the merge per PERF-1.
- `sync.Pool` reset functions zero the slice header before returning to the pool (avoid cross-iteration data leak).

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| PERF-1 (Bench gating) | `*_bench_test.go` files; CI runs `go test -bench=. -benchmem -count=10`; benchstat compares; >10% regression fails |
| PERF-2 (Cache-friendly) | Weights + activations stored as flat `[]T` slices owned by Network; index arithmetic instead of pointer chasing |
| PERF-3 (Bounded goroutines) | Batch-level worker pool sized to `runtime.GOMAXPROCS(0)`; per-neuron goroutines forbidden |
| PERF-4 (Zero/pooled inner loop) | `sync.Pool` for `[]T` activation buffers; preallocate at `Compile()` for everything sized by topology |
| PERF-5 (pprof opt-in) | `WithProfiling(":6060")` option starts `net/http.Server` with `net/http/pprof` registered; off by default |

## 5. Detailed Design

### 5.1 Buffer Pools

```go
// [REFERENCE] In pkg/network/pool.go.
var activationPool = sync.Pool{
    New: func() any { return make([]T, 0, 64) },
}

func acquireActivations[T utils.Float](size int) []T {
    buf := activationPool.Get().([]T)
    if cap(buf) < size { buf = make([]T, size) } else { buf = buf[:size] }
    return buf
}

func releaseActivations[T utils.Float](buf []T) {
    for i := range buf { buf[i] = 0 }   // zero before return
    activationPool.Put(buf[:0])
}
```

### 5.2 Compile()-time Preallocation

At `Compile()`:

```text
n.weightStorage = make([]T, totalAxonCount(topology))
n.activationStorage = make([]T, maxLayerSize(topology))
n.gradientStorage = make([]T, maxLayerSize(topology))
n.biasStorage = make([]T, totalBiasCount(topology))
```

Subsequent forward/backward passes reuse these — no allocation in the inner loop.

### 5.3 Worker Pool

```go
type workerPool struct {
    workers int
    jobs    chan job
    wg      sync.WaitGroup
}

func newWorkerPool() *workerPool {
    return &workerPool{workers: runtime.GOMAXPROCS(0)}
}
```

Used for batch-level parallelism (per-sample in a mini-batch); inactive on batchSize=1.

### 5.4 Profiling Hook

```go
func WithProfiling[T utils.Float](addr string) Option[T] {
    return func(c *Config[T]) { c.ProfilingAddr = addr }
}
```

At `Compile()`: if `ProfilingAddr != ""`, spawn `net/http.ListenAndServe(addr, nil)` in a goroutine. `pprof` registers handlers via blank import `_ "net/http/pprof"`.

### 5.5 Open Questions

- <!-- TBD: benchstat threshold — 10% per PERF-1, but noisy benchmarks may need 15%. Configure per-benchmark? -->
- <!-- TBD: SIMD/asm path location — `pkg/compute/cpu/asm_amd64.s`? Defer to backend spec. -->
- <!-- TBD: NUMA pinning — premature for v1. -->

## 6. Implementation Notes

1. New file `pkg/network/pool.go` for `sync.Pool` declarations.
2. Benchmarks adopt naming `Benchmark<Op>_<Topology>_<DType>` per PERF-1 table — e.g., `BenchmarkForward_XOR_f32`.
3. CI integration: a Make target `bench-compare` runs `benchstat base.txt pr.txt` and exits non-zero on regression.
4. `WithProfiling` lives in `pkg/nn/options.go` next to other functional options (l2-nn-facade).

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[POOL]` | `pkg/network/pool.go` (new) | sync.Pool declarations |
| `[BENCH]` | `pkg/*/bench_*_test.go` | Benchmark home (per PERF-1) |
| `[PROF]` | `pkg/nn/options.go` | `WithProfiling` option |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-28 | Initial Draft — concrete Go realization of l1-performance-contract RFC. |
| 1.0.0 | 2026-05-01 | [Batch-Stabilize] Draft → Stable. L1 parent now Stable. MVC satisfied: Overview + Invariant Compliance PERF-1..5 + pool/prealloc/worker-pool design. C9 Trust Mode. |
