// Package network — buffer pools and pre-allocated storage.
//
// Implements [l2-perf-impl] §5.1 (sync.Pool for transient activation
// buffers) and §5.2 (Compile-time preallocation hooks). The pools are
// per-precision globals; the generic AcquireActivations / ReleaseActivations
// functions dispatch to the matching pool at compile time via Go's
// type-switch on a zero value of T.
//
// Pool semantics per [l1-performance-contract] PERF-4:
//   - Buffers returned via Release are zeroed before being put back so
//     no caller observes stale data from a previous owner.
//   - Buffers smaller than requested are discarded and a fresh make()
//     of the requested size is returned (still pooled on Release).
package network

import (
	"sync"

	"github.com/teratron/gonn/pkg/utils"
)

// activationPoolF32 / activationPoolF64 hold reusable scratch slices
// keyed by element type. New returns a zero-length slice with a small
// capacity hint so first-use callers grow without reallocation in the
// common XOR-sized case.
var (
	activationPoolF32 = sync.Pool{
		New: func() any { return make([]float32, 0, 64) },
	}
	activationPoolF64 = sync.Pool{
		New: func() any { return make([]float64, 0, 64) },
	}
)

// AcquireActivations returns a []T of the requested size, drawing from
// the per-type pool. The returned slice is zeroed (Release clears
// before Put) so callers may rely on a clean baseline.
func AcquireActivations[T utils.Float](size int) []T {
	if size < 0 {
		size = 0
	}
	var z T
	switch any(z).(type) {
	case float32:
		buf := activationPoolF32.Get().([]float32)
		if cap(buf) < size {
			buf = make([]float32, size)
		} else {
			buf = buf[:size]
			for i := range buf {
				buf[i] = 0
			}
		}
		return any(buf).([]T)
	case float64:
		buf := activationPoolF64.Get().([]float64)
		if cap(buf) < size {
			buf = make([]float64, size)
		} else {
			buf = buf[:size]
			for i := range buf {
				buf[i] = 0
			}
		}
		return any(buf).([]T)
	default:
		return make([]T, size)
	}
}

// ReleaseActivations zeroes buf and parks it in the per-type pool so a
// subsequent Acquire of similar size avoids the allocator. Calling
// Release on a nil slice is safe — the function early-returns.
func ReleaseActivations[T utils.Float](buf []T) {
	if buf == nil {
		return
	}
	for i := range buf {
		buf[i] = 0
	}
	switch raw := any(buf).(type) {
	case []float32:
		activationPoolF32.Put(raw[:0])
	case []float64:
		activationPoolF64.Put(raw[:0])
	}
}

// PreallocStorage holds the flat buffers allocated once at Compile time
// per [l2-perf-impl] §5.2. Owning them on Network[T] keeps cache lines
// hot across forward / backward passes and guarantees zero allocation
// in the inner loop (PERF-4).
type PreallocStorage[T utils.Float] struct {
	WeightStorage     []T
	ActivationStorage []T
	GradientStorage   []T
	BiasStorage       []T
}

// NewPreallocStorage sizes each slice based on the topology dimensions.
// totalAxonCount is the sum of axons across all hidden+output layers;
// maxLayerSize is the widest layer (used for activation/gradient scratch);
// totalBiasCount is the count of bias-enabled output neurons across all
// bias-enabled layers.
func NewPreallocStorage[T utils.Float](totalAxonCount, maxLayerSize, totalBiasCount int) PreallocStorage[T] {
	return PreallocStorage[T]{
		WeightStorage:     make([]T, totalAxonCount),
		ActivationStorage: make([]T, maxLayerSize),
		GradientStorage:   make([]T, maxLayerSize),
		BiasStorage:       make([]T, totalBiasCount),
	}
}
