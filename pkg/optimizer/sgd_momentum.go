package optimizer

import (
	"encoding/json"

	"github.com/teratron/gonn/pkg/utils"
)

// SGDMomentum extends SGD with a velocity term: v = γ·v + lr·d; w -= v.
// Velocity slices are lazily preallocated on the first Step call.
//
// Default momentum coefficient γ = 0.9.
//
// AI-Meta:
//   - Purpose: SGD with momentum for faster convergence on ravine-shaped loss surfaces.
//   - Usage: opt := optimizer.NewSGDMomentum[float32](0.01); passed to nn.WithOptimizer.
//   - Concurrency: SingleGoroutine; Step mutates velocity slice in place.
//   - Related: [Optimizer], [NewSGD], [NewAdam].
//   - Stability: Stable.
type SGDMomentum[T utils.Float] struct {
	lr       T
	velocity []float64
	momentum float64
}

// compile-time interface verification (C26).
var _ Optimizer[float32] = (*SGDMomentum[float32])(nil)

// NewSGDMomentum returns an SGD-with-momentum optimizer (γ=0.9 default).
//
// AI-Meta:
//   - Purpose: Construct an SGD+momentum optimizer with the given learning rate.
//   - Usage: opt := optimizer.NewSGDMomentum[float32](0.01).
//   - Related: [SGDMomentum].
//   - Stability: Stable.
func NewSGDMomentum[T utils.Float](lr T) *SGDMomentum[T] {
	return &SGDMomentum[T]{lr: lr, momentum: 0.9}
}

// NewSGDMomentumWithGamma returns SGD+momentum with a custom momentum coefficient.
//
// AI-Meta:
//   - Purpose: Construct SGD+momentum with an explicit momentum coefficient γ.
//   - Usage: opt := optimizer.NewSGDMomentumWithGamma[float32](0.01, 0.95).
//   - Related: [SGDMomentum], [NewSGDMomentum].
//   - Stability: Stable.
func NewSGDMomentumWithGamma[T utils.Float](lr T, gamma float64) *SGDMomentum[T] {
	return &SGDMomentum[T]{lr: lr, momentum: gamma}
}

// Step applies: v[i] = γ·v[i] + lr·d[i]; w[i] -= v[i].
// Returns nil immediately when lr == 0 (OPT-3).
func (s *SGDMomentum[T]) Step(weights, deltas []T) error {
	if s.lr == 0 {
		return nil
	}
	if len(s.velocity) != len(weights) {
		if s.velocity != nil {
			utils.Logger.Warn("SGDMomentum.Step: weight count changed — resetting velocity",
				"old", len(s.velocity), "new", len(weights))
		}
		s.velocity = make([]float64, len(weights))
	}
	for i := range weights {
		s.velocity[i] = s.momentum*s.velocity[i] + float64(s.lr)*float64(deltas[i])
		weights[i] -= T(s.velocity[i])
	}
	return nil
}

// Reset zeroes the velocity buffer and resets internal state.
func (s *SGDMomentum[T]) Reset() {
	for i := range s.velocity {
		s.velocity[i] = 0
	}
}

// LearningRate returns the configured step size.
func (s *SGDMomentum[T]) LearningRate() T { return s.lr }

// SetLearningRate replaces the effective learning rate. Implements LearningRateSetter
// so that BindScheduler can push updated rates from a Scheduler.
func (s *SGDMomentum[T]) SetLearningRate(rate T) { s.lr = rate }

// SaveState serialises lr, momentum, and velocity to JSON.
func (s *SGDMomentum[T]) SaveState() ([]byte, error) {
	return json.Marshal(struct {
		Velocity []float64 `json:"velocity"`
		LR       float64   `json:"lr"`
		Momentum float64   `json:"momentum"`
	}{LR: float64(s.lr), Momentum: s.momentum, Velocity: s.velocity})
}

// LoadState restores from a SaveState blob.
func (s *SGDMomentum[T]) LoadState(data []byte) error {
	var st struct {
		Velocity []float64 `json:"velocity"`
		LR       float64   `json:"lr"`
		Momentum float64   `json:"momentum"`
	}
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	s.lr = T(st.LR)
	s.momentum = st.Momentum
	s.velocity = st.Velocity
	return nil
}
