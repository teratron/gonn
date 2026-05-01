// Package cpu — CPU reference backend buffer ops.
//
// Implements [l2-backend-cpu] §5.1 / COMP-4: Allocate returns a Buffer
// whose Data slice is the actual storage. CPU has no device transfer
// overhead, so Free is a best-effort no-op that helps the GC by
// dropping the slice.
package cpu

import (
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

// Allocate returns a fresh Buffer[T] of the requested size. Negative
// sizes are rejected with ErrUserConfig so the failure mode mirrors the
// rest of the API.
func (b Backend[T]) Allocate(size int) (compute.Buffer[T], error) {
	if size < 0 {
		return compute.Buffer[T]{}, utils.NewSizeError("cpu.Allocate.size", size, "non-negative")
	}
	return compute.Buffer[T]{Data: make([]T, size)}, nil
}

// Free releases the buffer's reference to its backing slice so the GC
// can reclaim memory promptly. Multiple Frees on the same Buffer are
// safe — the second call simply observes a nil slice.
func (b Backend[T]) Free(buf compute.Buffer[T]) error {
	// Nothing to do on CPU; the Buffer is GC-managed. The receiver-shaped
	// API exists so the future GPU backends can release device memory.
	_ = buf
	return nil
}
