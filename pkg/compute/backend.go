// Package compute — pluggable compute-backend interface.
//
// Implements [l1-compute-backend] §5.1 and [l2-backend-cpu] §5.2.
// The Backend[T] interface is the boundary between the math kernels
// (CPU reference, future OpenCL / CUDA) and the rest of the library.
// All cross-backend data transfer is explicit through Allocate / Free
// (COMP-4); no hidden copies, no implicit synchronisation.
package compute

import (
	"github.com/teratron/gonn/pkg/utils"
)

// LayerHandle is the per-layer view a compute backend needs to run the
// forward/backward kernels. Weights is shaped [outputSize][inputSize];
// Bias is per-output (empty when the layer has no bias). Only UpdateWeights
// is allowed to write through the weight/bias slices.
//
// AI-Meta:
//   - Purpose: Pass layer topology and activation functions to a backend kernel without exposing graph internals.
//   - Concurrency: NotSafe; only UpdateWeights may mutate Weights/Bias.
//   - Related: [Backend], [Buffer].
type LayerHandle[T utils.Float] struct {
	Activation func(T) T
	Derivative func(T) T
	Weights    [][]T
	Bias       []T
	Size       int
}

// Buffer is the explicit device-memory handle returned by Backend.Allocate.
// CPU backends embed the slice directly; future GPU backends would store a
// device pointer. Callers must release via Backend.Free.
//
// AI-Meta:
//   - Purpose: Opaque buffer handle for device memory; abstracts CPU slice vs. GPU pointer.
//   - Concurrency: NotSafe; caller owns the buffer exclusively between Allocate and Free.
//   - Related: [Backend], [LayerHandle].
type Buffer[T utils.Float] struct {
	Data []T
}

// Backend is the minimal surface every compute target must implement. The
// CPU reference path is always available as the default. Name() lets the
// registry route by string and identifies the active backend in logs.
//
// AI-Meta:
//   - Purpose: Pluggable compute boundary between math kernels and the rest of the library.
//   - Implementations: CPU reference backend (internal); future: OpenCL, CUDA.
//   - Concurrency: Depends on implementation; CPU backend is NotSafe by default.
//   - Related: [LayerHandle], [Buffer], [Register], [Get].
type Backend[T utils.Float] interface {
	Name() string
	Forward(layer LayerHandle[T], input []T) ([]T, error)
	Backward(layer LayerHandle[T], gradient []T) ([]T, error)
	UpdateWeights(layer LayerHandle[T], inputs, deltas []T, rate T) error
	Allocate(size int) (Buffer[T], error)
	Free(buf Buffer[T]) error
}
