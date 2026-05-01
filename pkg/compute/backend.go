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

// LayerHandle is the per-layer view a backend needs to evaluate the
// forward / backward kernels. Weights is shaped [outputSize][inputSize];
// Bias is per-output (empty when the layer has no bias). Activation is
// applied element-wise to the pre-activation sum.
//
// The handle is a value type — backends should not mutate Weights or
// Bias outside of UpdateWeights, which is the only kernel allowed to
// write through the slices.
type LayerHandle[T utils.Float] struct {
	Size       int
	Weights    [][]T
	Bias       []T
	Activation func(T) T
	Derivative func(T) T
}

// Buffer is the explicit handle returned by Backend.Allocate. CPU
// backends use the embedded slice directly; future GPU backends store
// a device pointer here. Per COMP-4 callers must release buffers via
// Backend.Free.
type Buffer[T utils.Float] struct {
	Data []T
}

// Backend is the small surface every compute target implements. The CPU
// reference path is the always-available default (COMP-1). Returning
// Name() lets the registry route by string and lets logs identify the
// active backend.
type Backend[T utils.Float] interface {
	Name() string
	Forward(layer LayerHandle[T], input []T) ([]T, error)
	Backward(layer LayerHandle[T], gradient []T) ([]T, error)
	UpdateWeights(layer LayerHandle[T], inputs, deltas []T, rate T) error
	Allocate(size int) (Buffer[T], error)
	Free(buf Buffer[T]) error
}
