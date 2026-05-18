# GPU Compute Backend

**Version:** 0.1.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-compute-backend.md

## Overview

Concrete Go realization of [l1-compute-backend.md](l1-compute-backend.md) for **GPU-accelerated
backends** — sibling to [l2-backend-cpu.md](l2-backend-cpu.md). Defines the `pkg/compute/gpu/`
umbrella package and its two cgo-isolated sub-packages: `pkg/compute/gpu/opencl/` (cross-vendor
OpenCL 1.2+) and `pkg/compute/gpu/cuda/` (NVIDIA CUDA 11+). Each sub-package registers a
`Backend[T]` factory under its name (`"opencl"`, `"cuda"`), is gated behind a build tag
(`cgo,opencl` / `cgo,cuda`) so default builds remain pure-Go, and adheres to the COMP-1 numerical
tolerance contract against the CPU reference.

The L1 parent explicitly anticipated this delivery (COMP-2 backend selection, COMP-3 fallback to
CPU, COMP-4 explicit transfer). This spec fixes Go-specific bindings — package paths, build tags,
device-buffer pinning, kernel granularity, and the cgo error-marshaling pattern — without
introducing any GPU types into the public `pkg/nn` API.

## Related Specifications

- [l1-compute-backend.md](l1-compute-backend.md) — Parent — invariants COMP-1..COMP-4 + interface sketch
- [l2-backend-cpu.md](l2-backend-cpu.md) — Sibling — reference implementation; numerical tolerance contract; registration pattern mirrored here
- [l2-perf-impl.md](l2-perf-impl.md) — Performance practices; GPU kernels are Layer 5 of the optimization stack (per l1-performance-contract.md)
- [l1-performance-contract.md](l1-performance-contract.md) — GPU is Layer 5; this spec inherits the perf benchmark contract
- [l2-errors-impl.md](l2-errors-impl.md) — Error sentinels — adds `ErrBackendUnavailable`, `ErrBackendTransfer`, `ErrBackendKernel`

## 1. Motivation

`l2-backend-cpu.md` is the always-on reference path; it is correct but slow on the matmul-heavy
hot loop. For modest networks (XOR, MNIST FC) this is fine. For Conv2D + MNIST CNN (Phase 14) the
forward / backward pass is already 80% kernel time — Phase 14's gate benchmarks measured 240 ms
per epoch on a 4-core CPU for 60k MNIST samples. A GPU backend reduces that to single-digit
milliseconds per epoch on commodity hardware.

GoNN must remain usable without a GPU (per L1 COMP-3 + C29 stdlib-only default). The build-tag
isolation here ensures `go build ./...` on a stock developer machine continues to compile pure-Go
and link only `pkg/compute/cpu/`. Opt-in is explicit: `go build -tags 'opencl'` or
`go build -tags 'cuda'` pulls in the cgo bindings.

The first delivery targets **OpenCL** (cross-vendor — AMD, Intel, NVIDIA on Linux/Windows/macOS
up to macOS 12) as the broader-reach baseline; **CUDA** follows as the higher-performance
NVIDIA-only path. Both share the device-side kernel sources via OpenCL C99 / CUDA C++ minimal
divergence (the kernel bodies are nearly identical; the host harness differs).

## 2. Constraints & Assumptions

- **cgo + build tags isolation**: `pkg/compute/gpu/opencl/` and `pkg/compute/gpu/cuda/` are the
  ONLY directories where cgo is permitted in the project. The umbrella `pkg/compute/gpu/` is
  pure-Go (constants, fallback shim).
- **Driver discovery is runtime**: `init()` does NOT call the driver; it registers a deferred
  factory. The driver handshake (`clGetPlatformIDs` / `cuInit`) happens in the factory invocation
  triggered by `WithBackend("opencl")` / `WithBackend("cuda")`. Missing driver → return
  `ErrBackendUnavailable`; the caller chain (per COMP-3) falls back to CPU with a `Warn` log.
- **Buffer is opaque**: `Buffer[T]` for GPU wraps the device handle (`cl_mem` / `CUdeviceptr`)
  plus the host-side length. No `.Slice()` accessor — host code that needs to read the data
  uses `backend.Read(buf, dst []T)` (explicit transfer per COMP-4).
- **No fp16 in v0.1.0**: `T utils.Float = float32 | float64` only. fp16 / mixed precision
  deferred to a future v0.2.0 amendment behind a separate spec.
- **Kernel granularity is per-layer**: `Forward(layer, input)` issues one kernel launch per layer
  (matmul + activation fused). Per-batch dispatch is the v0.2.0 target after we measure the
  current granularity.
- **Tolerance contract (COMP-1)**: GPU output must match CPU within `cpu.ToleranceF32` (1e-5) and
  `cpu.ToleranceF64` (1e-12). Tests cross-reference both backends on every kernel.
- **No build tag → no link**: a user without OpenCL / CUDA SDK installed does `go build ./...`
  and the GPU packages are excluded from compilation entirely — no link errors, no missing-symbol
  warnings, no driver checks.

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| COMP-1 (Reference impl tolerance) | GPU `cpu_test.go` cross-references every kernel against `pkg/compute/cpu/`; `pkg/compute/gpu/opencl/opencl_test.go` skips when no device, otherwise runs the full kernel matrix |
| COMP-2 (Compile-time selection) | `WithBackend("opencl")` / `WithBackend("cuda")` looks up the registered factory; factory invocation initialises the device and returns `Backend[T]` or `ErrBackendUnavailable` |
| COMP-3 (Fallback to CPU) | `pkg/nn/compile.go` wraps the backend factory call: on `ErrBackendUnavailable` it logs `Warn` via the `goLogger` and falls through to `cpu` factory; never silently produces wrong results — wrong-result paths return `ErrBackendKernel` |
| COMP-4 (Explicit transfer) | `Buffer[T]` is opaque; host ↔ device transfer goes through `Backend[T].Write(buf, src []T) error` and `Backend[T].Read(buf, dst []T) error`; no hidden copies in `Forward` / `Backward` |

## 5. Detailed Design

### 5.1 Package Layout

```text
pkg/compute/
├── backend.go              // unchanged — Backend[T] interface (l1-compute-backend §5.1)
├── registry.go             // unchanged — Register / Get factory by name
├── cpu/                    // unchanged — see l2-backend-cpu.md
│   └── ...
└── gpu/
    ├── doc.go              // pure-Go: package doc, vendor enum constants
    ├── unavailable.go      // pure-Go: shim that returns ErrBackendUnavailable on construct
    ├── opencl/
    │   ├── opencl.go       // build tag: cgo,opencl — Backend[T] impl
    │   ├── opencl_test.go  // build tag: cgo,opencl — cross-reference vs cpu
    │   ├── kernels.go      // host-side kernel program loaders
    │   ├── kernels.cl      // device-side OpenCL C99 source (embedded via go:embed)
    │   ├── buffer.go       // build tag: cgo,opencl — Buffer[T] wrapping cl_mem
    │   └── bindings.go     // cgo bindings — minimal OpenCL ICD loader
    └── cuda/
        ├── cuda.go         // build tag: cgo,cuda — Backend[T] impl
        ├── cuda_test.go    // build tag: cgo,cuda
        ├── kernels.go
        ├── kernels.cu      // device-side CUDA C++ source
        ├── buffer.go
        └── bindings.go     // cgo bindings — minimal libcuda.so loader
```

### 5.2 Build Tag Discipline

```go
//go:build cgo && opencl

package opencl

// ... cgo + OpenCL host code ...
```

The `cgo` half is mandatory — pure-Go builds skip these files entirely. The `opencl` / `cuda`
half is the opt-in toggle. A user without the SDK does not need to set either tag; the package is
simply excluded.

The shim `pkg/compute/gpu/unavailable.go` is pure-Go and always present so that callers can refer
to constant strings (`gpu.VendorOpenCL`, `gpu.VendorCUDA`) without conditional imports.

### 5.3 Backend Skeleton (OpenCL)

```go
// [REFERENCE] In pkg/compute/gpu/opencl/opencl.go.
//go:build cgo && opencl

package opencl

// #cgo LDFLAGS: -lOpenCL
// #include <CL/cl.h>
import "C"

import (
    _ "embed"

    "github.com/teratron/gonn/pkg/compute"
    "github.com/teratron/gonn/pkg/utils"
)

//go:embed kernels.cl
var kernelsSource string

type Backend[T utils.Float] struct {
    ctx     C.cl_context
    queue   C.cl_command_queue
    program C.cl_program
    // ... cached kernel handles ...
}

func (b *Backend[T]) Name() string { return "opencl" }

func (b *Backend[T]) Forward(layer compute.LayerHandle, input compute.Buffer[T]) (compute.Buffer[T], error) {
    // 1. Enqueue clSetKernelArg for layer.Weights, layer.Bias, input handle, output handle.
    // 2. clEnqueueNDRangeKernel with global = layer.Size, local chosen per device.
    // 3. Return the output buffer handle (no host read — caller decides when to Read).
}

// Backward, UpdateWeights, Allocate, Free, Read, Write — analogous.
```

### 5.4 Registration

```go
// [REFERENCE] init() side-effect at package opencl.
//go:build cgo && opencl

package opencl

func init() {
    compute.Register("opencl", func[T utils.Float]() (compute.Backend[T], error) {
        b, err := initialise[T]()
        if err != nil {
            return nil, fmt.Errorf("opencl: %w", utils.ErrBackendUnavailable)
        }
        return b, nil
    })
}
```

The factory signature is `func() (Backend[T], error)` — an error is returned ONLY for
`ErrBackendUnavailable`. Kernel errors during Forward/Backward propagate as `ErrBackendKernel`
through the standard return path.

The blank import `_ "github.com/teratron/gonn/pkg/compute/gpu/opencl"` is added by the user
explicitly when they want OpenCL — it is NOT added by `pkg/nn/init.go` (unlike `cpu`).

### 5.5 Fallback Wiring in `pkg/nn/compile.go`

```go
// [REFERENCE] In pkg/nn/compile.go (adjacent to existing backend resolution).
func (n *NN[T]) resolveBackend() (compute.Backend[T], error) {
    if n.cfg.Backend == "" || n.cfg.Backend == "cpu" {
        return compute.MustGet[T]("cpu"), nil
    }
    b, err := compute.Get[T](n.cfg.Backend)
    if errors.Is(err, utils.ErrBackendUnavailable) {
        slog.Warn("backend unavailable, falling back to cpu", "requested", n.cfg.Backend, "err", err)
        return compute.MustGet[T]("cpu"), nil
    }
    return b, err
}
```

The `Warn`-then-fallback flow is the COMP-3 contract verbatim. `ErrBackendKernel` (runtime
kernel failure) is NOT caught here — it propagates to the caller of `Train` / `Query` so the
user sees a real failure rather than a silent slowdown.

### 5.6 Error Sentinels

```go
// [REFERENCE] Additions to pkg/utils/errors.go.
var (
    ErrBackendUnavailable = errors.New("compute backend unavailable")
    ErrBackendTransfer    = errors.New("compute backend transfer failed")
    ErrBackendKernel      = errors.New("compute backend kernel failed")
)
```

All three wrap the `ErrCompute` category sentinel introduced by the same patch
(`l1-error-taxonomy.md` adds `Compute` as a new category alongside `IO`, `Config`, `Integrity`).

### 5.7 Kernel Source Embedding

OpenCL kernels (`kernels.cl`) and CUDA kernels (`kernels.cu`) are embedded via `//go:embed` so
the compiled binary is self-contained. No runtime file lookup. The kernel source itself is
plain-text C99 / CUDA C++ — readable in source review and version-controlled.

```go
//go:embed kernels.cl
var kernelsSource string
```

### 5.8 Testing Strategy

| Test | Build tag | Purpose |
| :--- | :--- | :--- |
| `pkg/compute/gpu/unavailable_test.go` | (none) | Pure-Go: confirms shim returns `ErrBackendUnavailable` |
| `pkg/compute/gpu/opencl/opencl_test.go` | `cgo,opencl` | Cross-references every kernel vs CPU within tolerance; SKIPS if `clGetPlatformIDs` returns 0 devices |
| `pkg/compute/gpu/cuda/cuda_test.go` | `cgo,cuda` | Same pattern, CUDA-side |

CI matrix: default builds run pure-Go tests + `cpu` backend (existing); a separate CI runner with
OpenCL ICD installed runs `go test -tags 'opencl' ./...`; CUDA path is optional, gated on a
self-hosted runner with an NVIDIA GPU.

## 6. Implementation Notes

1. **Phase A** — `pkg/compute/gpu/` umbrella package + `unavailable.go` shim + sentinel additions to `pkg/utils/errors.go`. No cgo. Validates the build-tag discipline (pure-Go default).
2. **Phase B** — `pkg/compute/gpu/opencl/` skeleton: cgo binding, `Allocate` / `Free` / `Write` / `Read`. Test against CPU for buffer round-trip (no kernel yet).
3. **Phase C** — `opencl` Forward kernel for Dense layer (matmul + bias + activation fused); cross-reference vs CPU. Conv1D / Conv2D / Pool kernels follow once Dense lands.
4. **Phase D** — `opencl` Backward + UpdateWeights. End-to-end XOR training on `opencl` backend; convergence parity vs CPU.
5. **Phase E** — `pkg/nn/compile.go` fallback wiring + Warn-log path; `TestBackendFallback` test in `pkg/nn`.
6. **Phase F** — `pkg/compute/gpu/cuda/` mirror of Phases B-D. Shares kernel source where syntax overlaps; diverges where CUDA C++ specifics demand.
7. **Phase G** — Performance benchmark suite: MNIST CNN epoch-time on cpu vs opencl vs cuda. Gate criterion: opencl ≥ 5× cpu on a desktop GPU; cuda ≥ 10× cpu on a desktop GPU.

Phase G's gate is the **promotion criterion** from v0.1.0 (this spec) to v1.0.0 — measured
speedup is the proof that the backend abstraction actually delivers on its motivation.

## 7. Drawbacks & Alternatives

- **Alternative: pure-Go via SIMD intrinsics (Avo / asmdecl)** — rejected; significant complexity, gains 2-4× over scalar Go but not 10-100× like GPU. Worth revisiting as a separate Layer-2-of-optimization spec (perf-impl already at Layer 1-4).
- **Alternative: WebGPU via Wasm** — rejected; WebGPU host bindings in Go are immature, and the target deployment is not browser-side.
- **Alternative: single combined `pkg/compute/gpu/` with runtime driver selection** — rejected; cgo demands compile-time linkage to either libOpenCL or libcuda (not both). Build-tag separation matches that constraint.
- **Vulkan compute** — deferred to v0.3.0+ once OpenCL / CUDA paths are validated. Vulkan offers vendor-neutrality + lower overhead than OpenCL but its compute API surface is larger.

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[L1-COMP]` | `.design/main/specifications/l1-compute-backend.md` | Parent contract — COMP-1..COMP-4 invariants are the input to §4 |
| `[CPU-REF]` | `.design/main/specifications/l2-backend-cpu.md` | Sibling — registration pattern, tolerance constants, kernel granularity baseline |
| `[CPU-PKG]` | `pkg/compute/cpu/` | Reference implementation in Go — cross-reference target for COMP-1 |
| `[BACKEND-IFACE]` | `pkg/compute/backend.go` | The `Backend[T]` interface this spec implements |
| `[REGISTRY]` | `pkg/compute/registry.go` | Factory registration mechanism reused verbatim |
| `[ERR-PKG]` | `pkg/utils/errors.go` | `ErrBackendUnavailable`, `ErrBackendTransfer`, `ErrBackendKernel` sentinel additions |
| `[PERF-CONTRACT]` | `.design/main/specifications/l1-performance-contract.md` | Layer 5 (backend) of the optimization stack |
| `[FLT]` | `pkg/utils/float.go` | `Float = float32 \| float64` constraint — fp16 deferred |

<!-- Downstream agent instruction: §5.2 build-tag discipline is normative. Do NOT add cgo to any
     package outside pkg/compute/gpu/opencl/ and pkg/compute/gpu/cuda/. The umbrella pkg/compute/gpu/
     is pure-Go by design. -->

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 0.1.0 | 2026-05-18 | Initial spec authored via `/magic-spec` Spark 2. `pkg/compute/gpu/` umbrella + `opencl/` + `cuda/` sub-packages, build-tag isolation (`cgo,opencl` / `cgo,cuda`), three new error sentinels (`ErrBackendUnavailable`, `ErrBackendTransfer`, `ErrBackendKernel`), 7-phase implementation plan. Promoted Draft → Stable via Trust Mode (MVC: Overview + §4 Invariant Compliance + §5 Detailed Design; no RULES conflicts; no cycles; L1 parent `l1-compute-backend.md` Stable v1.0.0). |
