package optimizer

import (
	"encoding/json"
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// ExponentialLR decays the learning rate by a constant factor gamma on every
// step. Formula: lr = lr₀ × gamma^t, where t is the global step counter.
// After each Step() call the counter increments by one and the new rate is
// returned. Granularity defaults to PerEpoch (LRS-5).
//
// AI-Meta:
//   - Purpose: Exponential-decay LR scheduler — multiply rate by gamma every step.
//   - Usage: sched := optimizer.NewExponentialLR[float32](0.1, 0.95); wire via BindScheduler.
//   - Lifecycle: Step advances the counter; Reset restores step 0 and lr₀.
//   - Related: [Scheduler], [BindScheduler], [StepLR], [CosineAnnealingLR].
//   - Stability: Stable.
type ExponentialLR[T utils.Float] struct {
	current T           // most recently computed rate
	gran    Granularity // granularity of step calls
	lr0     T           // initial learning rate (lr₀)
	gamma   float64     // multiplicative decay factor per step
	step    uint        // global step counter
}

// compile-time interface verification (C26).
var _ Scheduler[float32] = (*ExponentialLR[float32])(nil)

// NewExponentialLR returns an ExponentialLR scheduler.
// lr0 is the initial rate; gamma is the per-step decay factor (0 < gamma ≤ 1).
// Granularity defaults to PerEpoch.
//
// AI-Meta:
//   - Purpose: Construct an exponential-decay scheduler with lr₀ and per-step gamma.
//   - Usage: sched := optimizer.NewExponentialLR[float64](0.01, 0.9).
//   - Related: [ExponentialLR], [BindScheduler].
//   - Stability: Stable.
func NewExponentialLR[T utils.Float](lr0 T, gamma float64) *ExponentialLR[T] {
	return &ExponentialLR[T]{
		lr0:     lr0,
		current: lr0,
		gamma:   gamma,
		gran:    PerEpoch,
	}
}

// NewExponentialLRWithGranularity returns an ExponentialLR scheduler with
// explicit granularity.
//
// AI-Meta:
//   - Purpose: Construct ExponentialLR with explicit PerEpoch or PerStep granularity.
//   - Related: [ExponentialLR], [NewExponentialLR], [Granularity].
//   - Stability: Stable.
func NewExponentialLRWithGranularity[T utils.Float](lr0 T, gamma float64, g Granularity) *ExponentialLR[T] {
	e := NewExponentialLR(lr0, gamma)
	e.gran = g
	return e
}

// Step advances the schedule by one step and returns the new rate.
// Formula: lr₀ × gamma^t.
func (e *ExponentialLR[T]) Step() T {
	e.step++
	e.current = T(float64(e.lr0) * math.Pow(e.gamma, float64(e.step)))
	return e.current
}

// Reset restores the scheduler to its initial state (step 0, rate lr₀).
func (e *ExponentialLR[T]) Reset() {
	e.step = 0
	e.current = e.lr0
}

// Granularity reports the step-call frequency.
func (e *ExponentialLR[T]) Granularity() Granularity { return e.gran }

// exponentialLRState is the JSON-serialisable snapshot of ExponentialLR.
type exponentialLRState struct {
	LR0     float64 `json:"lr0"`
	Current float64 `json:"current"`
	Gamma   float64 `json:"gamma"`
	Step    uint    `json:"step"`
	Gran    uint8   `json:"gran"`
}

// SaveState serialises all ExponentialLR state to JSON (LRS-6).
func (e *ExponentialLR[T]) SaveState() ([]byte, error) {
	return json.Marshal(exponentialLRState{
		LR0:     float64(e.lr0),
		Current: float64(e.current),
		Gamma:   e.gamma,
		Step:    e.step,
		Gran:    uint8(e.gran),
	})
}

// LoadState restores from a SaveState blob.
func (e *ExponentialLR[T]) LoadState(data []byte) error {
	var st exponentialLRState
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	e.lr0 = T(st.LR0)
	e.current = T(st.Current)
	e.gamma = st.Gamma
	e.step = st.Step
	e.gran = Granularity(st.Gran)
	return nil
}
