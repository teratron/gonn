package recurrent

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

// Compile-time check: utils.Orthogonal must exist for T=float64 (REC-7 dependency).
var _ = utils.Orthogonal[float64]

// applySigmoidFused applies sigmoid activation in-place to gates[offset:offset+count].
func applySigmoidFused[T utils.Float](gates []T, offset, count int) {
	for i := offset; i < offset+count; i++ {
		gates[i] = activation.Activation[T](gates[i], activation.SIGMOID)
	}
}

// applyTanhFused applies tanh activation in-place to gates[offset:offset+count].
func applyTanhFused[T utils.Float](gates []T, offset, count int) {
	for i := offset; i < offset+count; i++ {
		gates[i] = activation.Activation[T](gates[i], activation.TanH)
	}
}

// initHiddenCache allocates a zeroed slice of length (seqLen+1)*hidden for the
// BPTT hidden-state cache. Index 0 stores h_0 (initial state = zeros).
func initHiddenCache[T utils.Float](seqLen, hidden int) []T {
	return make([]T, (seqLen+1)*hidden)
}

// matVecAdd computes dst[i] += Σ_j mat[i*cols+j]*vec[j] for every i in [0,rows).
// mat is [rows, cols] row-major; dst must be pre-allocated to at least rows elements
// and is modified in-place (callers zero it before the first call).
func matVecAdd[T utils.Float](dst, mat, vec []T, rows, cols int) {
	for i := range rows {
		base := i * cols
		var acc T
		for j := range cols {
			acc += mat[base+j] * vec[j]
		}
		dst[i] += acc
	}
}
