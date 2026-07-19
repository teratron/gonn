// Package gpu provides the GPU compute-backend umbrella for GoNN.
//
// Implements [l2-backend-gpu] §5.1–§5.2. This package is always importable
// (pure Go, no cgo). Sub-packages expose concrete backends gated by build
// tags:
//
//   - pkg/compute/gpu/opencl — requires "cgo" and "opencl" build tags.
//   - pkg/compute/gpu/cuda  — requires "cgo" and "cuda" build tags.
//
// When neither sub-package is registered, [New] returns
// [utils.ErrBackendUnavailable] and the caller (pkg/nn compile()) falls
// back to the CPU backend with a Warn-level log message (COMP-3).
//
// AI-Meta:
//   - Purpose: GPU compute-backend umbrella; exposes New[T] for vendor resolution and VendorXxx constants.
//   - Usage: gpu.New[float32](gpu.VendorOpenCL) — returns Backend or ErrBackendUnavailable for CPU fallback.
//   - Concurrency: Safe; New is a read-only registry lookup.
//   - Related: [compute.Backend], [compute.Register], [utils.ErrBackendUnavailable].
//   - Stability: Stable.
package gpu

// VendorOpenCL is the registry key for the OpenCL backend.
//
// AI-Meta:
//   - Purpose: String key identifying the OpenCL compute backend in the registry.
//   - Usage: gpu.New[float64](gpu.VendorOpenCL).
//   - Related: [VendorCUDA], [New].
//   - Stability: Stable.
const VendorOpenCL = "opencl"

// VendorCUDA is the registry key for the CUDA compute backend.
//
// AI-Meta:
//   - Purpose: String key identifying the CUDA compute backend in the registry.
//   - Usage: gpu.New[float64](gpu.VendorCUDA).
//   - Related: [VendorOpenCL], [New].
//   - Stability: Stable.
const VendorCUDA = "cuda"
