package regularizer

import "github.com/teratron/gonn/pkg/utils"

// compose holds a slice of member regularizers.
type compose[T utils.Float] struct {
	members []Regularizer[T]
}

// compile-time interface verification (C26).
var _ Regularizer[float32] = (*compose[float32])(nil)

// Compose returns a Regularizer that combines the members additively:
// Penalty sums each member's penalty; ApplyMask chains them left-to-right (REG-6).
//
// AI-Meta:
//   - Purpose: Combine multiple regularizers into one; e.g., Compose(L2(0.01), Dropout(0.8)).
//   - Usage: reg := regularizer.Compose[float32](regularizer.NewL2[float32](0.01), regularizer.NewDropout[float32](0.8)).
//   - Related: [Regularizer], [NewL2], [NewL1], [NewDropout].
//   - Stability: Stable.
func Compose[T utils.Float](rs ...Regularizer[T]) Regularizer[T] {
	members := make([]Regularizer[T], 0, len(rs))
	for _, r := range rs {
		if r != nil {
			members = append(members, r)
		}
	}
	return &compose[T]{members: members}
}

// Penalty sums all member penalties (REG-6).
func (c *compose[T]) Penalty(weights []T) T {
	var total T
	for _, r := range c.members {
		total += r.Penalty(weights)
	}
	return total
}

// WeightGrad sums each member's per-weight gradient contribution (REG-6).
func (c *compose[T]) WeightGrad(w T) T {
	var total T
	for _, r := range c.members {
		total += r.WeightGrad(w)
	}
	return total
}

// ApplyMask chains members left-to-right, passing the output of each as input
// to the next (REG-6). Returns acts unchanged when the member list is empty.
func (c *compose[T]) ApplyMask(acts []T, training bool) []T {
	for _, r := range c.members {
		acts = r.ApplyMask(acts, training)
	}
	return acts
}
