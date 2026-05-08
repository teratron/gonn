package regularizer

import "github.com/teratron/gonn/pkg/utils"

// L1 applies L1 (lasso) regularization: penalty = λ × Σ|wᵢ|.
// ApplyMask is an identity — L1 does not modify activations.
//
// AI-Meta:
//   - Purpose: L1 lasso regularizer; promotes sparse weight distributions.
//   - Usage: reg := regularizer.NewL1[float32](0.01); passed to nn.WithRegularizer.
//   - Related: [Regularizer], [NewL2], [Compose].
//   - Stability: Stable.
type L1[T utils.Float] struct {
	lambda T
}

// compile-time interface verification (C26).
var _ Regularizer[float32] = (*L1[float32])(nil)

// NewL1 returns an L1 regularizer with penalty coefficient λ.
//
// AI-Meta:
//   - Purpose: Construct an L1 regularizer; λ=0.01 is a common starting point.
//   - Usage: reg := regularizer.NewL1[float32](0.01).
//   - Related: [L1].
//   - Stability: Stable.
func NewL1[T utils.Float](lambda T) *L1[T] {
	return &L1[T]{lambda: lambda}
}

// Penalty returns λ × Σ|wᵢ|.
func (r *L1[T]) Penalty(weights []T) T {
	var sum T
	for _, w := range weights {
		if w < 0 {
			sum -= w
		} else {
			sum += w
		}
	}
	return r.lambda * sum
}

// ApplyMask returns acts unchanged — L1 has no mask effect.
func (r *L1[T]) ApplyMask(acts []T, _ bool) []T { return acts }
