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
//   - Concurrency: NotSafe.
//   - Related: [LayerHandle], [Buffer], [Register], [Get].
type Backend[T utils.Float] interface {
	Name() string
	Forward(layer LayerHandle[T], input []T) ([]T, error)
	Backward(layer LayerHandle[T], gradient []T) ([]T, error)
	UpdateWeights(layer LayerHandle[T], inputs, deltas []T, rate T) error
	Allocate(size int) (Buffer[T], error)
	Free(buf Buffer[T]) error
}

// DenseMatrix is the structure-of-arrays view of one fully connected layer:
// a single contiguous weight run in row-major [Out][In] order, so slot (o, j)
// lives at W[o*In+j]. When the layer has a bias, it occupies the LAST input
// column (input[In-1] is pinned to 1) — kernels therefore never special-case
// bias, they just multiply one wider matrix.
//
// AI-Meta:
//   - Purpose: Contiguous matrix view of a dense layer passed to accelerated kernels.
//   - Concurrency: NotSafe; the caller owns W for the duration of the call.
//   - Related: [DenseKernels], [Backend].
//   - Stability: Stable.
type DenseMatrix[T utils.Float] struct {
	W   []T
	In  int
	Out int
}

// DenseKernels is the OPTIONAL accelerated linear-algebra path a [Backend] may
// implement. The training engine type-asserts for it: a backend that provides
// these three primitives has its kernels driven for every forward and backward
// pass, and one that does not falls back to the engine's internal reference
// loops — identical math, simply not delegated.
//
// The split is deliberate. These are pure primitives with no notion of
// activations, losses, optimizers, normalization, or dropout: that orchestration
// stays in the engine, so plugging in a backend can never silently bypass the
// configured optimizer or regularizer. Only the inner products move.
//
// AI-Meta:
//   - Purpose: Optional capability interface letting a backend supply the dense matrix primitives.
//   - Implementations: [cpu.Backend]; OpenCL when built with the opencl tag.
//   - Usage: type-asserted by pkg/network; absence is not an error.
//   - Concurrency: NotSafe; one call at a time per layer.
//   - Related: [DenseMatrix], [Backend].
//   - Stability: Stable.
type DenseKernels[T utils.Float] interface {
	// MatVec computes preact[o] = Σ_j m.W[o*In+j] * input[j].
	MatVec(m DenseMatrix[T], input, preact []T) error

	// MatVecT computes dInput[j] = Σ_o delta[o] * m.W[o*In+j] — the transpose
	// product that carries the error signal to the previous layer.
	MatVecT(m DenseMatrix[T], delta, dInput []T) error

	// GradOuter computes grad[o*In+j] = scale * delta[o] * input[j], the
	// per-weight gradient of one sample. scale lets the caller fold in a sign
	// convention without a second pass over the array.
	GradOuter(m DenseMatrix[T], delta, input, grad []T, scale T) error
}
