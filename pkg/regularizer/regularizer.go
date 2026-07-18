// Package regularizer — generalization constraints for GoNN training loops.
//
// Every concrete type implements [Regularizer[T]]. Apply is a nil-safe helper
// that wraps the hot-path calls with a nil check so callers do not need to
// guard at each call site.
package regularizer

import "github.com/teratron/gonn/pkg/utils"

// Regularizer adds a generalization constraint to the training loop.
// Penalty adds the regularization term to the effective loss; ApplyMask
// optionally drops or scales activations (Dropout) during the forward pass.
//
// AI-Meta:
//   - Purpose: Contract for regularization strategies; plug into pkg/nn via WithRegularizer.
//   - Usage: Pass a concrete implementation to nn.WithRegularizer[float32](regularizer.NewL2[float32](0.01)).
//   - Lifecycle: Created before Compile; Penalty called once per sample; ApplyMask after forward pass.
//   - Concurrency: SingleGoroutine; implementations may hold mutable RNG state.
//   - Related: [Apply], [NewL1], [NewL2], [NewDropout], [Compose].
//   - Stability: Stable.
type Regularizer[T utils.Float] interface {
	// Penalty returns the regularization term added to the loss for this sample.
	// weights is the flat weight slice from the network.
	Penalty(weights []T) T

	// WeightGrad returns the regularization contribution to ∂L/∂w for a single
	// weight w: L2 → 2λw, L1 → λ·sign(w), Dropout → 0. The training loop adds
	// this to the data gradient before the optimizer step, so the penalty
	// actually decays the weights instead of only inflating the reported loss
	// (audit B/L2: regularization previously never reached the gradient).
	WeightGrad(w T) T

	// ApplyMask optionally modifies activations in-place.
	// training=true: Dropout samples the mask; L1/L2 return acts unchanged.
	// training=false: all implementations are strict no-ops (REG-3).
	ApplyMask(activations []T, training bool) []T
}

// Apply is a nil-safe wrapper around Regularizer.ApplyMask and Regularizer.Penalty.
// When reg is nil it returns activations unchanged and a zero penalty.
// Callers in the training loop use this to avoid guarding at every call site.
//
// AI-Meta:
//   - Purpose: Nil-safe entry point for regularizer calls inside the training loop.
//   - Usage: acts = regularizer.Apply(n.reg, acts, training); penalty = regularizer.Penalty(n.reg, weights).
//   - Related: [Regularizer].
//   - Stability: Stable.
func Apply[T utils.Float](reg Regularizer[T], acts []T, training bool) []T {
	if reg == nil {
		return acts
	}
	return reg.ApplyMask(acts, training)
}

// Penalty is a nil-safe wrapper that returns zero when reg is nil.
//
// AI-Meta:
//   - Purpose: Nil-safe penalty accessor; returns 0 when no regularizer is configured.
//   - Related: [Regularizer], [Apply].
//   - Stability: Stable.
func Penalty[T utils.Float](reg Regularizer[T], weights []T) T {
	if reg == nil {
		return 0
	}
	return reg.Penalty(weights)
}

// AddWeightGrad adds the regularizer's per-weight gradient contribution into
// grads in place (grads[i] += reg.WeightGrad(weights[i])). Nil-safe no-op when
// reg is nil. Called by the training loop after the data gradient is computed
// and before the optimizer step.
//
// AI-Meta:
//   - Purpose: Fold L1/L2 weight-decay into the gradient so regularization affects training.
//   - Related: [Regularizer.WeightGrad], [Penalty].
//   - Stability: Stable.
func AddWeightGrad[T utils.Float](reg Regularizer[T], weights, grads []T) {
	if reg == nil {
		return
	}
	n := min(len(weights), len(grads))
	for i := 0; i < n; i++ {
		grads[i] += reg.WeightGrad(weights[i])
	}
}
