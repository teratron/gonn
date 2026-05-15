package optimizer

import (
	"encoding/json"
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// StepLR decays the learning rate by a constant factor gamma every stepSize
// steps. Formula: lr = lr₀ × gamma^(⌊t / stepSize⌋). After schedule
// completion the last computed rate is held constant (LRS-4).
//
// AI-Meta:
//   - Purpose: Step-decay LR scheduler — multiply rate by gamma every N steps.
//   - Usage: sched := optimizer.NewStepLR[float32](0.1, 10, 0.5); wire via BindScheduler.
//   - Lifecycle: Step advances the counter; Reset restores step 0 and lr₀.
//   - Related: [Scheduler], [BindScheduler], [WarmUpLR], [CosineAnnealingLR].
//   - Stability: Stable.
type StepLR[T utils.Float] struct {
	current  T // most recently computed rate
	gran     Granularity
	lr0      T       // initial learning rate (lr₀)
	gamma    float64 // decay factor per interval
	step     uint    // global step counter
	stepSize uint    // interval length in steps
}

// compile-time interface verification (C26).
var _ Scheduler[float32] = (*StepLR[float32])(nil)

// NewStepLR returns a StepLR scheduler.
// lr0 is the initial rate, stepSize is the decay interval, gamma is the
// multiplicative factor (0 < gamma ≤ 1). Granularity defaults to PerEpoch.
//
// AI-Meta:
//   - Purpose: Construct a step-decay scheduler with lr₀, interval, and decay factor.
//   - Usage: sched := optimizer.NewStepLR[float64](0.1, 10, 0.5).
//   - Related: [StepLR], [BindScheduler].
//   - Stability: Stable.
func NewStepLR[T utils.Float](lr0 T, stepSize uint, gamma float64) *StepLR[T] {
	return &StepLR[T]{
		lr0:      lr0,
		current:  lr0,
		gamma:    gamma,
		stepSize: stepSize,
		gran:     PerEpoch,
	}
}

// NewStepLRWithGranularity returns a StepLR scheduler with explicit granularity.
//
// AI-Meta:
//   - Purpose: Construct StepLR with explicit PerEpoch or PerStep granularity.
//   - Related: [StepLR], [NewStepLR], [Granularity].
//   - Stability: Stable.
func NewStepLRWithGranularity[T utils.Float](lr0 T, stepSize uint, gamma float64, g Granularity) *StepLR[T] {
	s := NewStepLR(lr0, stepSize, gamma)
	s.gran = g
	return s
}

// Step advances the schedule by one interval and returns the new rate.
// Formula: lr₀ × gamma^(⌊t / stepSize⌋).
func (s *StepLR[T]) Step() T {
	s.step++
	interval := s.step / s.stepSize
	s.current = T(float64(s.lr0) * math.Pow(s.gamma, float64(interval)))
	return s.current
}

// Reset restores the scheduler to its initial state (step 0, rate lr₀).
func (s *StepLR[T]) Reset() {
	s.step = 0
	s.current = s.lr0
}

// Granularity reports the step-call frequency.
func (s *StepLR[T]) Granularity() Granularity { return s.gran }

// stepLRState is the JSON-serialisable snapshot of StepLR.
type stepLRState struct {
	LR0      float64 `json:"lr0"`
	Current  float64 `json:"current"`
	Gamma    float64 `json:"gamma"`
	StepSize uint    `json:"step_size"`
	Step     uint    `json:"step"`
	Gran     uint8   `json:"gran"`
}

// SaveState serialises all StepLR state to JSON (LRS-6).
func (s *StepLR[T]) SaveState() ([]byte, error) {
	return json.Marshal(stepLRState{
		LR0:      float64(s.lr0),
		Current:  float64(s.current),
		Gamma:    s.gamma,
		StepSize: s.stepSize,
		Step:     s.step,
		Gran:     uint8(s.gran),
	})
}

// LoadState restores from a SaveState blob.
func (s *StepLR[T]) LoadState(data []byte) error {
	var st stepLRState
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	s.lr0 = T(st.LR0)
	s.current = T(st.Current)
	s.gamma = st.Gamma
	s.stepSize = st.StepSize
	s.step = st.Step
	s.gran = Granularity(st.Gran)
	return nil
}
