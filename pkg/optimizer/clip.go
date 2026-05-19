package optimizer

import (
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// ClipByGlobalNorm clips a set of gradient slices so that the global L2 norm
// ||g||₂ does not exceed threshold (REC-7, BPTT stability).
//
// The scale factor is min(1, threshold / ||g||₂); when the norm is already ≤
// threshold the gradients are returned unchanged. All slices are modified
// in-place. A threshold ≤ 0 is a no-op.
//
// AI-Meta:
//   - Purpose: Clip gradient slices in-place to keep the global L2 norm ≤ threshold (REC-7).
//   - Usage: optimizer.ClipByGlobalNorm(grads, 1.0) before opt.Step.
//   - Concurrency: NotSafe; slices are mutated.
//   - Related: [WithGradClipNorm].
//   - Stability: Stable.
func ClipByGlobalNorm[T utils.Float](grads [][]T, threshold T) {
	if threshold <= 0 || len(grads) == 0 {
		return
	}
	var sumSq float64
	for _, g := range grads {
		for _, v := range g {
			f := float64(v)
			sumSq += f * f
		}
	}
	norm := math.Sqrt(sumSq)
	if norm == 0 || norm <= float64(threshold) {
		return
	}
	scale := T(float64(threshold) / norm)
	for _, g := range grads {
		for i := range g {
			g[i] *= scale
		}
	}
}
