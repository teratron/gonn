package regularizer

import "github.com/teratron/gonn/pkg/utils"

// L2 applies L2 (weight-decay) regularization: penalty = λ × Σwᵢ².
// ApplyMask is an identity — L2 does not modify activations.
//
// AI-Meta:
//   - Purpose: L2 weight-decay regularizer; penalises large weights to reduce overfitting.
//   - Usage: reg := regularizer.NewL2[float32](0.01); passed to nn.WithRegularizer.
//   - Related: [Regularizer], [NewL1], [Compose].
//   - Stability: Stable.
type L2[T utils.Float] struct {
	lambda T
}

// compile-time interface verification (C26).
var _ Regularizer[float32] = (*L2[float32])(nil)

// NewL2 returns an L2 regularizer with penalty coefficient λ.
//
// AI-Meta:
//   - Purpose: Construct an L2 regularizer; λ=0.01 is a common starting point.
//   - Usage: reg := regularizer.NewL2[float32](0.01).
//   - Related: [L2].
//   - Stability: Stable.
func NewL2[T utils.Float](lambda T) *L2[T] {
	return &L2[T]{lambda: lambda}
}

// Penalty returns λ × Σwᵢ².
func (r *L2[T]) Penalty(weights []T) T {
	var sum T
	for _, w := range weights {
		sum += w * w
	}
	return r.lambda * sum
}

// WeightGrad returns 2λw, the L2 contribution to ∂L/∂w (weight decay).
func (r *L2[T]) WeightGrad(w T) T { return 2 * r.lambda * w }

// ApplyMask returns acts unchanged — L2 has no mask effect.
func (r *L2[T]) ApplyMask(acts []T, _ bool) []T { return acts }
