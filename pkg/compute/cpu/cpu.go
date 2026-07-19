// Package cpu — reference CPU backend for GoNN.
//
// Implements [l2-backend-cpu] (Stable v1.0.0). The CPU backend is the
// always-available default (COMP-1) and the numerical reference every
// other backend must match within ToleranceF32 / ToleranceF64. Pure Go,
// no cgo, no SIMD — extension points live in kernels.go.
//
// Importing this package for side effects registers "cpu" with both
// float32 and float64 registries:
//
//	import _ "github.com/teratron/gonn/pkg/compute/cpu"
//
// pkg/nn imports it transitively so end-users get the default behaviour
// without thinking about it.
package cpu

import (
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/utils"
)

// Tolerance constants for cross-backend comparisons (COMP-1). Other
// backends' tests use these to assert their kernel outputs match the
// CPU reference within tolerance.
const (
	ToleranceF32 float32 = 1e-5
	ToleranceF64 float64 = 1e-12
)

// Name identifies this backend in the registry and in error messages.
const Name = "cpu"

// Backend is the zero-state CPU backend. It carries no fields because
// CPU kernels operate directly on caller-owned slices — there is no
// device handle to track.
type Backend[T utils.Float] struct{}

// Name returns the backend's identifier; satisfies compute.Backend.
func (b Backend[T]) Name() string { return Name }

func init() {
	compute.Register(Name, func() compute.Backend[float32] {
		return Backend[float32]{}
	})
	compute.Register(Name, func() compute.Backend[float64] {
		return Backend[float64]{}
	})
}

// Compile-time interface verification per C26.
var (
	_ compute.Backend[float32] = (*Backend[float32])(nil)
	_ compute.Backend[float64] = (*Backend[float64])(nil)
)
