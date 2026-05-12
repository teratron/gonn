// Package norm provides normalization layers for the GoNN neural network library.
//
// Implements the Normalizer[T] interface with three concrete types —
// BatchNorm[T], LayerNorm[T], and GroupNorm[T] — satisfying NORM-1..9
// from l1-normalization-layers. Affine parameters (γ, β) participate in
// the optimizer cycle via GradSlots, identical to Dense weight updates.
package norm

import (
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// NormMode selects the operational mode for normalization layers.
// BatchNorm branches on mode to choose between batch statistics (Train)
// and running statistics (Eval). LayerNorm and GroupNorm are mode-agnostic.
//
// AI-Meta:
//   - Purpose: Enum selecting train-time batch stats vs eval-time running stats for BatchNorm.
//   - Usage: layer.SetMode(norm.NormTrain) before training, SetMode(norm.NormEval) for inference.
//   - Related: [Normalizer], [BatchNorm], [SetMode].
//   - Stability: Stable.
type NormMode int32

const (
	// NormTrain uses batch mean/variance and updates running statistics (EMA).
	NormTrain NormMode = 0
	// NormEval uses frozen running mean/variance; no EMA update.
	NormEval NormMode = 1
)

// Normalizer is the common interface implemented by BatchNorm[T], LayerNorm[T],
// and GroupNorm[T]. It satisfies the same shape contract as other layer types
// (InputSize/OutputSize/Forward) so it is composable with the existing layer stack.
//
// GradSlots returns the gradient-accumulation slices for the affine parameters
// gamma and beta. Returns (nil, nil) when affine is disabled. The training loop
// passes these alongside Dense weight gradients to the optimizer step.
//
// AI-Meta:
//   - Purpose: Common interface for normalization layers; composable with the existing layer hierarchy.
//   - Usage: var _ Normalizer[float32] = (*BatchNorm[float32])(nil).
//   - Implementations: [BatchNorm], [LayerNorm], [GroupNorm].
//   - Concurrency: SetMode is goroutine-safe via atomic; Forward must be called from a single goroutine.
//   - Related: [BatchNorm], [LayerNorm], [GroupNorm], [NormMode].
//   - Stability: Stable.
type Normalizer[T utils.Float] interface {
	// Forward applies normalization to x and returns a new slice of the same length.
	Forward(x []T) []T
	// SetMode switches between NormTrain and NormEval. Safe to call concurrently.
	SetMode(m NormMode)
	// GradSlots returns the gradient-accumulation slices for gamma and beta.
	// Returns (nil, nil) when affine is disabled (identity path).
	GradSlots() (gamma, beta []T)
	// InputSize returns the expected input feature count.
	InputSize() int
	// OutputSize always equals InputSize; output shape is preserved (NORM-1).
	OutputSize() int
}

// Compile-time interface assertions — catch missing methods at build time (C26).
var _ Normalizer[float32] = (*BatchNorm[float32])(nil)
var _ Normalizer[float32] = (*LayerNorm[float32])(nil)
var _ Normalizer[float32] = (*GroupNorm[float32])(nil)

// stddev computes the standard deviation from pre-computed variance and an
// epsilon floor that prevents division by zero (NORM-4).
//
// AI-Meta:
//   - Purpose: Shared helper computing standard deviation with epsilon guard for all norm types.
//   - Usage: sd := stddev(variance, eps) inside Forward implementations.
//   - Concurrency: Safe; pure function, no state.
func stddev[T utils.Float](variance, eps T) T {
	return T(math.Sqrt(float64(variance+eps)))
}

// applyAffine scales each element of xHat by gamma[i] and adds beta[i].
// When both slices are nil (affine disabled) the input is returned unchanged.
//
// AI-Meta:
//   - Purpose: Shared affine transform applied after normalization when gamma/beta are registered.
//   - Usage: out = applyAffine(xHat, gamma, beta) inside Forward implementations.
//   - Concurrency: Safe; pure function operating on caller-provided slices.
func applyAffine[T utils.Float](xHat, gamma, beta []T) []T {
	if gamma == nil && beta == nil {
		return xHat
	}
	out := make([]T, len(xHat))
	for i, v := range xHat {
		out[i] = v
		if gamma != nil {
			out[i] *= gamma[i]
		}
		if beta != nil {
			out[i] += beta[i]
		}
	}
	return out
}
