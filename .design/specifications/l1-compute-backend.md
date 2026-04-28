# Compute Backend Abstraction

**Version:** 0.1.0
**Status:** RFC
**Layer:** concept

## Overview

Defines a pluggable compute-backend interface so GoNN can offload heavy linear algebra (matrix
multiply, activation, gradient accumulation) to GPU accelerators (OpenCL, CUDA) without leaking
GPU types into the public API. The default CPU backend is pure Go; alternative backends are opt-in
via build tags or runtime registration.

## Related Specifications

- [l1-neural-network-architecture.md](l1-neural-network-architecture.md) — Parent
- [l1-performance-contract.md](l1-performance-contract.md) — Backend is Layer 5 of the optimization stack

## 1. Motivation

Modern NN training is bottlenecked by matrix operations. CPUs hit a wall; GPUs are 10–100× faster
for these kernels. But GoNN must remain **usable without a GPU** — a hard dependency on CUDA / OpenCL
would alienate the majority of users.

The answer is a **backend interface**: `pkg/compute.Backend` with a default in-process Go
implementation and pluggable alternatives loaded via cgo / build tags. The library code calls
backend methods; user picks the backend at `Compile()` time.

## 2. Constraints & Assumptions

> **Concept-Layer Notation**: References to `cgo`, build tags, and Go sub-package paths are
> **illustrative bindings** of universal concepts (foreign-function interface, conditional compilation,
> module isolation) to the project's Go implementation. The backend contract — interface, fallback
> rule, numerical tolerance — is language-independent.

- The interface must be **small** — adding a backend requires implementing ~5 methods, not 50.
- The default CPU backend is **always present** — zero external dependencies (per `C29`).
- GPU backends use the language's **foreign-function interface** (Go: cgo) and live in **isolated
  sub-modules** so users without the GPU runtime can ignore them entirely.
- **Conditional compilation** (Go: build tags `//go:build cgo && opencl`) keeps the default build pure-Go.

## 3. Core Invariants

- **COMP-1**: The CPU backend is the **reference implementation**. Numerical results from any other
  backend must match within a documented tolerance (e.g., `1e-5` for `float32`).
- **COMP-2**: Backend selection is fixed at `Compile()`-time. Switching mid-run is forbidden.
- **COMP-3**: Backend errors **fall back to CPU** with a `Warn` log when the GPU runtime is missing
  at startup; never silently produce wrong results.
- **COMP-4**: All cross-backend data transfer (host ↔ device) is **explicit** — no hidden copies.
  The interface returns handles, not slices.

## 5. Detailed Design

### 5.1 Backend Interface (proposed)

```text
type Backend[T utils.Float] interface {
    Name() string
    Forward(layer LayerHandle, input []T) ([]T, error)
    Backward(layer LayerHandle, gradient []T) ([]T, error)
    UpdateWeights(layer LayerHandle, rate T) error
    Allocate(size int) (Buffer[T], error)
    Free(Buffer[T]) error
}
```

### 5.2 Concrete Backends (planned)

| Backend | Build tag | Status |
| :--- | :--- | :--- |
| `cpu` | (always) | Default, MUST |
| `opencl` | `cgo,opencl` | Future, MAY |
| `cuda` | `cgo,cuda` | Future, MAY |

### 5.3 Open Questions

- <!-- TBD: kernel granularity — per-layer, per-batch, or per-network forward call? -->
- <!-- TBD: pinned host memory for fast transfers — backend-internal -->
- <!-- TBD: fp16 / mixed precision — extension of T, separate spec when needed -->

## 6. Implementation Notes

1. Phase 1: extract current inline matrix ops into `pkg/compute/cpu/` implementing `Backend[T]`.
2. Phase 2: route all `Network[T]` math through the interface (no functional change).
3. Phase 3: add `WithBackend(name string)` builder/option.
4. Phase 4: prototype `opencl` backend in a separate branch; benchmark vs CPU on a target workload.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[COMPUTE]` | `pkg/compute/` (new) | Backend interface and `cpu` reference implementation |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-04-27 | Initial Draft from TODO #13. |
| 0.1.0 | 2026-04-28 | Concept-Layer Notation added. Status promoted Draft → RFC. Backend interface and 4 invariants ready for review. |
