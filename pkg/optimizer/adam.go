package optimizer

import (
	"encoding/json"
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// Adam implements the Adam optimizer (Kingma & Ba 2014).
// First and second moment slices are lazily preallocated on the first Step call
// so construction is zero-alloc. Subsequent calls reuse the slices (0 allocs/op).
//
// Default hyper-parameters: β₁=0.9, β₂=0.999, ε=1e-8.
//
// AI-Meta:
//   - Purpose: Adam optimizer with adaptive per-weight learning rates; suitable for deep or sparse networks.
//   - Usage: opt := optimizer.NewAdam[float32](0.001); passed to nn.WithOptimizer.
//   - Concurrency: SingleGoroutine; Step mutates moment slices in place.
//   - Related: [Optimizer], [NewSGD], [NewRMSProp].
//   - Stability: Stable.
type Adam[T utils.Float] struct {
	lr                T
	beta1, beta2, eps float64
	t                 int
	m                 []float64 // first moment
	v                 []float64 // second moment
}

// compile-time interface verification (C26).
var _ Optimizer[float32] = (*Adam[float32])(nil)

// NewAdam returns an Adam optimizer with the given learning rate and default
// hyperparameters (β₁=0.9, β₂=0.999, ε=1e-8).
//
// AI-Meta:
//   - Purpose: Construct an Adam optimizer for gradient-adaptive training.
//   - Usage: opt := optimizer.NewAdam[float64](0.001).
//   - Related: [Adam], [NewAdam].
//   - Stability: Stable.
func NewAdam[T utils.Float](lr T) *Adam[T] {
	return &Adam[T]{
		lr:    lr,
		beta1: 0.9,
		beta2: 0.999,
		eps:   1e-8,
	}
}

// NewAdamHyper returns Adam with custom hyperparameters.
//
// AI-Meta:
//   - Purpose: Construct an Adam optimizer with explicit β₁, β₂, ε overrides.
//   - Usage: opt := optimizer.NewAdamHyper[float32](0.001, 0.9, 0.999, 1e-8).
//   - Related: [Adam], [NewAdam].
//   - Stability: Stable.
func NewAdamHyper[T utils.Float](lr T, beta1, beta2, eps float64) *Adam[T] {
	return &Adam[T]{lr: lr, beta1: beta1, beta2: beta2, eps: eps}
}

// Step applies the Adam update rule. Lazily allocates moment slices on first call.
// Returns nil immediately when lr == 0 (OPT-3).
func (a *Adam[T]) Step(weights, deltas []T) error {
	if a.lr == 0 {
		return nil
	}
	n := len(weights)
	if a.m == nil {
		a.m = make([]float64, n)
		a.v = make([]float64, n)
	}
	a.t++
	b1t := math.Pow(a.beta1, float64(a.t))
	b2t := math.Pow(a.beta2, float64(a.t))
	lr := float64(a.lr) * math.Sqrt(1-b2t) / (1 - b1t)
	for i := range weights {
		g := float64(deltas[i])
		a.m[i] = a.beta1*a.m[i] + (1-a.beta1)*g
		a.v[i] = a.beta2*a.v[i] + (1-a.beta2)*g*g
		weights[i] -= T(lr * a.m[i] / (math.Sqrt(a.v[i]) + a.eps))
	}
	return nil
}

// Reset zeroes moment slices and the step counter.
func (a *Adam[T]) Reset() {
	for i := range a.m {
		a.m[i] = 0
		a.v[i] = 0
	}
	a.t = 0
}

// LearningRate returns the base step size.
func (a *Adam[T]) LearningRate() T { return a.lr }

// SetLearningRate replaces the effective learning rate. Implements LearningRateSetter
// so that BindScheduler can push updated rates from a Scheduler.
func (a *Adam[T]) SetLearningRate(rate T) { a.lr = rate }

// adamState is the JSON-serialisable snapshot of Adam internal state.
type adamState struct {
	LR    float64   `json:"lr"`
	Beta1 float64   `json:"beta1"`
	Beta2 float64   `json:"beta2"`
	Eps   float64   `json:"eps"`
	T     int       `json:"t"`
	M     []float64 `json:"m"`
	V     []float64 `json:"v"`
}

// SaveState serialises lr, hyperparameters, t, m, v to JSON (OPT-6).
func (a *Adam[T]) SaveState() ([]byte, error) {
	return json.Marshal(adamState{
		LR:    float64(a.lr),
		Beta1: a.beta1,
		Beta2: a.beta2,
		Eps:   a.eps,
		T:     a.t,
		M:     a.m,
		V:     a.v,
	})
}

// LoadState restores from a SaveState blob.
func (a *Adam[T]) LoadState(data []byte) error {
	var st adamState
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	a.lr = T(st.LR)
	a.beta1 = st.Beta1
	a.beta2 = st.Beta2
	a.eps = st.Eps
	a.t = st.T
	a.m = st.M
	a.v = st.V
	return nil
}
