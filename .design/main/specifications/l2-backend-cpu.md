# CPU Compute Backend

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-compute-backend.md

## Overview

Concrete Go realization of [l1-compute-backend.md](l1-compute-backend.md) for the **default CPU backend**: the `pkg/compute/cpu/` package, the `Backend[T]` interface implementation, the `Buffer[T]` handle (a thin wrapper over `[]T`), and the registration mechanism that makes `cpu` the always-present zero-dependency reference backend.

## Related Specifications

- [l1-compute-backend.md](l1-compute-backend.md) — Parent — invariants COMP-1..COMP-4 + interface sketch
- [l2-perf-impl.md](l2-perf-impl.md) — Performance practices apply inside the CPU kernels
- [l2-network-graph.md](l2-network-graph.md) — Caller of backend forward/backward methods

## 1. Motivation

L1 defines the backend interface as language-neutral. This spec is the **reference implementation** that every other backend (OpenCL, CUDA) must match within COMP-1's tolerance. It fixes Go specifics — package path, build-tag-free always-on registration, kernel granularity, and the buffer-handle shape that future GPU backends will mirror.

## 2. Constraints & Assumptions

- Stdlib only — no cgo, no SIMD intrinsics in v1 (per `C29` and l1-performance-contract.md Layer 1–4 first).
- `Buffer[T]` for CPU is just `[]T` under the hood; no pinning, no device transfer overhead.
- Backend is registered at package init via `compute.Register("cpu", newCPU[T])` — always available, no build tag needed.
- Numerical results are the **golden reference**: every other backend's tests compare against CPU output within `1e-5` for `float32`, `1e-12` for `float64`.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| COMP-1 (Reference impl) | This spec defines the canonical math; tolerance constants exported as `cpu.ToleranceF32`, `cpu.ToleranceF64` |
| COMP-2 (Compile-time selection) | `WithBackend("cpu")` (default) wires the registered factory into `Network[T]` at `Compile()` |
| COMP-3 (Fallback to CPU) | N/A for CPU itself; CPU is the fallback target, never falls back further |
| COMP-4 (Explicit transfer) | `Allocate` returns `Buffer[T]{slice: make([]T, size)}`; no implicit copy on Forward/Backward — kernels read/write the same slice |

## 5. Detailed Design

### 5.1 Package Layout

```text
pkg/compute/
├── backend.go      // Backend[T] interface (per L1 §5.1)
├── registry.go     // Register / Get factory by name
└── cpu/
    ├── cpu.go          // Backend[T] impl
    ├── kernels.go      // Forward / Backward / UpdateWeights kernels
    ├── buffer.go       // Buffer[T] = struct { data []T }
    └── cpu_test.go
```

### 5.2 Backend Implementation Skeleton

```go
// [REFERENCE] In pkg/compute/cpu/cpu.go.
package cpu

type Backend[T utils.Float] struct{}

func (b Backend[T]) Name() string { return "cpu" }

func (b Backend[T]) Forward(layer LayerHandle, input []T) ([]T, error) {
    out := make([]T, layer.Size)
    for i := 0; i < layer.Size; i++ {
        var sum T
        for j, w := range layer.Weights[i] {
            sum += input[j] * w
        }
        out[i] = layer.Activation(sum + layer.Bias[i])
    }
    return out, nil
}

// Backward, UpdateWeights, Allocate, Free — analogous straightforward implementations.
```

### 5.3 Registration

```go
// [REFERENCE] init() side-effect at package compute/cpu.
func init() {
    compute.Register("cpu", func[T utils.Float]() compute.Backend[T] {
        return Backend[T]{}
    })
}
```

The blank import `_ "github.com/teratron/gonn/pkg/compute/cpu"` lives in `pkg/nn/init.go` so the default backend is always registered when the public API is used.

### 5.4 Tolerance Constants

```go
const (
    ToleranceF32 float32 = 1e-5
    ToleranceF64 float64 = 1e-12
)
```

Used by alternative-backend test suites for reference comparison (COMP-1).

### 5.5 Open Questions

- <!-- TBD: kernel granularity — current sketch is per-layer; consider per-batch fusion if benchmarks show overhead. -->
- <!-- TBD: SIMD entry — add `pkg/compute/cpu/asm_amd64.s` later, behind `//go:build amd64 && simd`. Out of v1 scope. -->
- <!-- TBD: float64 path — is there a use case beyond reference comparison? Probably keep both since utils.Float = float32 | float64. -->

## 6. Implementation Notes

1. New package `pkg/compute/` with `cpu/` subpackage.
2. Existing inline matrix ops in `pkg/network/propagation.go` are extracted into kernels and routed through the backend interface (per l1 §6 Phase 1–2).
3. No build tags on CPU — it must compile in every environment per COMP-1.
4. Tests compare against a hand-written numpy-style reference for a small fixed input — golden file under `pkg/compute/cpu/testdata/`.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[CPU]` | `pkg/compute/cpu/` (new) | Reference CPU backend |
| `[BACKEND]` | `pkg/compute/backend.go` (new) | Interface declaration |
| `[REGISTRY]` | `pkg/compute/registry.go` (new) | Factory registry |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-28 | Initial Draft — concrete Go realization of l1-compute-backend RFC, CPU reference path. |
| 1.0.0 | 2026-05-01 | [Batch-Stabilize] Draft → Stable. L1 parent now Stable. MVC satisfied: Overview + Invariant Compliance COMP-1..4 + Backend implementation skeleton. C9 Trust Mode. |
