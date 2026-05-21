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
		New: func() any { s := make([]float32, 0, 64); return &s },
	}
	activationPoolF64 = sync.Pool{
		New: func() any { s := make([]float64, 0, 64); return &s },
	}
)

// AcquireActivations returns a zeroed []T of the requested size from the
// per-precision pool. If the pooled buffer is too small, a fresh allocation
// is returned and the pool is bypassed.
//
// AI-Meta:
//   - Purpose: Allocate (or reuse) a scratch slice for activation values; reduces GC pressure in inner loops.
//   - Usage: buf := AcquireActivations[float32](n); defer ReleaseActivations(buf).
//   - Concurrency: Safe; pool is goroutine-safe via sync.Pool.
//   - Related: [ReleaseActivations], [PreallocStorage].
func AcquireActivations[T utils.Float](size int) []T {
	if size < 0 {
		size = 0
	}
	var z T
	switch any(z).(type) {
	case float32:
		buf := *activationPoolF32.Get().(*[]float32)
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
		buf := *activationPoolF64.Get().(*[]float64)
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

// ReleaseActivations zeroes buf and returns it to the per-precision pool.
// Calling Release on a nil or empty slice is a no-op.
//
// AI-Meta:
//   - Purpose: Return a scratch slice to the pool after use; must pair with each AcquireActivations call.
//   - Concurrency: Safe; pool is goroutine-safe via sync.Pool.
//   - Related: [AcquireActivations].
func ReleaseActivations[T utils.Float](buf []T) {
	if buf == nil {
		return
	}
	for i := range buf {
		buf[i] = 0
	}
	switch raw := any(buf).(type) {
	case []float32:
		tmp := raw[:0]
		activationPoolF32.Put(&tmp)
	case []float64:
		tmp := raw[:0]
		activationPoolF64.Put(&tmp)
	}
}

// PreallocStorage holds flat buffers allocated once at compile/build time
// to eliminate inner-loop heap allocations during forward/backward passes.
// Sized to the network topology by NewPreallocStorage.
//
// AI-Meta:
//   - Purpose: Zero-allocation scratch storage for weights, activations, gradients, and biases.
//   - Concurrency: NotSafe; owned by a single Network and mutated during forward/backward passes.
//   - Related: [NewPreallocStorage], [AcquireActivations].
type PreallocStorage[T utils.Float] struct {
	WeightStorage     []T
	ActivationStorage []T
	GradientStorage   []T
	BiasStorage       []T
}

// NewPreallocStorage allocates all four flat buffers sized to the given topology.
// totalAxonCount: sum of all axon counts across hidden + output layers.
// maxLayerSize: width of the widest layer (for activation/gradient scratch).
// totalBiasCount: number of bias-enabled neurons across all bias-enabled layers.
//
// AI-Meta:
//   - Purpose: Allocate topology-sized scratch buffers once; called during network compilation.
//   - Usage: s := network.NewPreallocStorage[float32](axons, maxWidth, biases).
//   - Related: [PreallocStorage].
func NewPreallocStorage[T utils.Float](totalAxonCount, maxLayerSize, totalBiasCount int) PreallocStorage[T] {
	return PreallocStorage[T]{
		WeightStorage:     make([]T, totalAxonCount),
		ActivationStorage: make([]T, maxLayerSize),
		GradientStorage:   make([]T, maxLayerSize),
		BiasStorage:       make([]T, totalBiasCount),
	}
}
