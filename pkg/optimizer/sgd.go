package optimizer

import (
	"encoding/json"

	"github.com/teratron/gonn/pkg/utils"
)

// SGD implements vanilla stochastic gradient descent: w[i] -= lr × d[i].
// Stateless — no moment buffers — so Step is always 0 allocs/op.
//
// AI-Meta:
//   - Purpose: Vanilla SGD optimizer; the package default returned by DefaultOptimizer.
//   - Usage: opt := optimizer.NewSGD[float32](0.3); passed to nn.WithOptimizer.
//   - Concurrency: SingleGoroutine; Step mutates the caller-supplied weights slice.
//   - Related: [Optimizer], [NewSGDMomentum], [NewAdam], [DefaultOptimizer].
//   - Stability: Stable.
type SGD[T utils.Float] struct {
	lr T
}

// compile-time interface verification (C26).
var _ Optimizer[float32] = (*SGD[float32])(nil)

// NewSGD returns an SGD optimizer with the given learning rate.
//
// AI-Meta:
//   - Purpose: Construct an SGD optimizer; use for simple or baseline training runs.
//   - Usage: opt := optimizer.NewSGD[float32](0.01).
//   - Related: [SGD], [DefaultOptimizer].
//   - Stability: Stable.
func NewSGD[T utils.Float](lr T) *SGD[T] {
	return &SGD[T]{lr: lr}
}

// Step applies: weights[i] -= lr × deltas[i] for each index.
// Returns nil immediately when lr == 0 (OPT-3).
func (s *SGD[T]) Step(weights, deltas []T) error {
	if s.lr == 0 {
		return nil
	}
	for i := range weights {
		weights[i] -= s.lr * deltas[i]
	}
	return nil
}

// Reset is a no-op for SGD — it carries no accumulated state.
func (s *SGD[T]) Reset() {}

// LearningRate returns the configured step size.
func (s *SGD[T]) LearningRate() T { return s.lr }

// SetLearningRate replaces the effective learning rate. Implements LearningRateSetter
// so that BindScheduler can push updated rates from a Scheduler.
func (s *SGD[T]) SetLearningRate(rate T) { s.lr = rate }

// SaveState serialises the learning rate to JSON.
func (s *SGD[T]) SaveState() ([]byte, error) {
	return json.Marshal(struct {
		LR float64 `json:"lr"`
	}{LR: float64(s.lr)})
}

// LoadState restores a saved learning rate.
func (s *SGD[T]) LoadState(data []byte) error {
	var st struct {
		LR float64 `json:"lr"`
	}
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	s.lr = T(st.LR)
	return nil
}
