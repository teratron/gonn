// Package optimizer — pluggable weight-update strategies for GoNN.
//
// Every concrete type implements [Optimizer[T]], a minimal interface that
// decouples the update rule from the training loop. The default is plain
// SGD, obtained via [DefaultOptimizer].
package optimizer

import "github.com/teratron/gonn/pkg/utils"

// Optimizer updates weight parameters in-place given their raw gradients.
// Step is called once per weight-update cycle (after CalculateMisses) with
// two parallel flat slices produced by the network's gradient collector:
//
//   - weights: current values — mutated in-place.
//   - deltas: ∂L/∂w for each weight — consumed read-only.
//
// AI-Meta:
//   - Purpose: Contract for weight-update strategies; plug into pkg/nn via WithOptimizer.
//   - Usage: Pass a concrete implementation to nn.WithOptimizer[float32](optimizer.NewAdam[float32](0.001)).
//   - Lifecycle: Created before Compile; Step called per training sample; Reset clears state between runs.
//   - Concurrency: SingleGoroutine; Step must not be called concurrently.
//   - Errors: ErrUserConfig on invalid configuration (e.g., negative learning rate).
//   - Related: [DefaultOptimizer], [NewSGD], [NewAdam], [NewSGDMomentum], [NewRMSProp].
//   - Stability: Stable.
type Optimizer[T utils.Float] interface {
	// Step updates weights in-place: weights[i] = f(weights[i], deltas[i], state).
	// Returns a non-nil error only on structural misuse (e.g., length mismatch).
	Step(weights, deltas []T) error

	// Reset zeroes all accumulated moment slices and the step counter.
	// Call before re-training on a different dataset.
	Reset()

	// LearningRate returns the base step size configured at construction.
	LearningRate() T

	// SaveState serialises internal optimiser state (moment slices, step counter)
	// to JSON so checkpoints can restore training exactly.
	SaveState() ([]byte, error)

	// LoadState restores from a SaveState blob. Returns an error if the blob
	// is malformed or was produced by a different optimiser type.
	LoadState([]byte) error
}

// DefaultOptimizer returns a vanilla SGD optimizer with the given learning rate.
// Used by compile() when the caller does not supply WithOptimizer.
//
// AI-Meta:
//   - Purpose: Fallback optimizer when none is configured; preserves pre-optimizer training behaviour.
//   - Usage: Resolved automatically by compile(); rarely called directly.
//   - Related: [Optimizer], [NewSGD].
//   - Stability: Stable.
func DefaultOptimizer[T utils.Float](lr T) Optimizer[T] {
	return NewSGD(lr)
}
