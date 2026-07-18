package optimizer

import (
	"encoding/json"
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// RMSProp maintains a moving average of squared gradients and scales the
// learning rate accordingly: E[g²][i] = α·E[g²][i] + (1-α)·g[i]²;
// w[i] -= lr / (√E[g²][i] + ε) × g[i].
//
// Default decay α=0.99, ε=1e-8. Squared-EMA slice is lazily allocated.
//
// AI-Meta:
//   - Purpose: RMSProp optimizer; adapts learning rate per weight using gradient magnitude history.
//   - Usage: opt := optimizer.NewRMSProp[float32](0.001); passed to nn.WithOptimizer.
//   - Concurrency: SingleGoroutine; Step mutates squared-EMA slice in place.
//   - Related: [Optimizer], [NewAdam], [NewSGD].
//   - Stability: Stable.
type RMSProp[T utils.Float] struct {
	lr     T
	sqGrad []float64
	alpha  float64
	eps    float64
}

// compile-time interface verification (C26).
var _ Optimizer[float32] = (*RMSProp[float32])(nil)

// NewRMSProp returns an RMSProp optimizer with the given learning rate and
// default hyperparameters (α=0.99, ε=1e-8).
//
// AI-Meta:
//   - Purpose: Construct an RMSProp optimizer; good default for recurrent networks.
//   - Usage: opt := optimizer.NewRMSProp[float32](0.001).
//   - Related: [RMSProp].
//   - Stability: Stable.
func NewRMSProp[T utils.Float](lr T) *RMSProp[T] {
	return &RMSProp[T]{lr: lr, alpha: 0.99, eps: 1e-8}
}

// NewRMSPropHyper returns RMSProp with custom α and ε.
//
// AI-Meta:
//   - Purpose: Construct an RMSProp optimizer with explicit α and ε overrides.
//   - Usage: opt := optimizer.NewRMSPropHyper[float32](0.001, 0.99, 1e-8).
//   - Related: [RMSProp], [NewRMSProp].
//   - Stability: Stable.
func NewRMSPropHyper[T utils.Float](lr T, alpha, eps float64) *RMSProp[T] {
	return &RMSProp[T]{lr: lr, alpha: alpha, eps: eps}
}

// Step applies the RMSProp update rule.
// Returns nil immediately when lr == 0 (OPT-3).
func (r *RMSProp[T]) Step(weights, deltas []T) error {
	if r.lr == 0 {
		return nil
	}
	if len(r.sqGrad) != len(weights) {
		if r.sqGrad != nil {
			utils.Logger.Warn("RMSProp.Step: weight count changed — resetting state",
				"old", len(r.sqGrad), "new", len(weights))
		}
		r.sqGrad = make([]float64, len(weights))
	}
	lr := float64(r.lr)
	for i := range weights {
		g := float64(deltas[i])
		r.sqGrad[i] = r.alpha*r.sqGrad[i] + (1-r.alpha)*g*g
		weights[i] -= T(lr / (math.Sqrt(r.sqGrad[i]) + r.eps) * g)
	}
	return nil
}

// Reset zeroes the squared-gradient EMA buffer.
func (r *RMSProp[T]) Reset() {
	for i := range r.sqGrad {
		r.sqGrad[i] = 0
	}
}

// LearningRate returns the configured step size.
func (r *RMSProp[T]) LearningRate() T { return r.lr }

// SetLearningRate replaces the effective learning rate. Implements LearningRateSetter
// so that BindScheduler can push updated rates from a Scheduler.
func (r *RMSProp[T]) SetLearningRate(rate T) { r.lr = rate }

// SaveState serialises lr, hyperparameters, and squared-EMA buffer to JSON.
func (r *RMSProp[T]) SaveState() ([]byte, error) {
	return json.Marshal(struct {
		SqGrad []float64 `json:"sq_grad"`
		LR     float64   `json:"lr"`
		Alpha  float64   `json:"alpha"`
		Eps    float64   `json:"eps"`
	}{LR: float64(r.lr), Alpha: r.alpha, Eps: r.eps, SqGrad: r.sqGrad})
}

// LoadState restores from a SaveState blob.
func (r *RMSProp[T]) LoadState(data []byte) error {
	var st struct {
		SqGrad []float64 `json:"sq_grad"`
		LR     float64   `json:"lr"`
		Alpha  float64   `json:"alpha"`
		Eps    float64   `json:"eps"`
	}
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	r.lr = T(st.LR)
	r.alpha = st.Alpha
	r.eps = st.Eps
	r.sqGrad = st.SqGrad
	return nil
}
