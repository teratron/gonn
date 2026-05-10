package optimizer

import (
	"encoding/json"

	"github.com/teratron/gonn/pkg/utils"
)

// WarmUpLR linearly ramps the learning rate from 0 to lr₀ over warmupSteps
// steps. After the warm-up period the rate is held constant at lr₀ (LRS-4).
// Formula: lr₀ × (t / warmupSteps) for t < warmupSteps; lr₀ after.
//
// AI-Meta:
//   - Purpose: Linear warm-up scheduler — ramps LR from zero to lr₀ over warmupSteps.
//   - Usage: sched := optimizer.NewWarmUpLR[float32](0.1, 1000); wire via BindScheduler.
//   - Lifecycle: Step advances counter; after warmupSteps holds lr₀; Reset restores step 0.
//   - Related: [Scheduler], [BindScheduler], [StepLR], [ChainScheduler].
//   - Stability: Stable.
type WarmUpLR[T utils.Float] struct {
	lr0         T    // target learning rate after warm-up
	warmupSteps uint // number of warm-up steps
	step        uint // global step counter
	gran        Granularity
}

// compile-time interface verification (C26).
var _ Scheduler[float32] = (*WarmUpLR[float32])(nil)

// NewWarmUpLR returns a WarmUpLR scheduler with PerStep granularity (the natural
// default for a warm-up phase that ramps per batch).
//
// AI-Meta:
//   - Purpose: Construct a linear warm-up scheduler; PerStep granularity by default.
//   - Usage: sched := optimizer.NewWarmUpLR[float64](0.1, 1000).
//   - Related: [WarmUpLR], [BindScheduler].
//   - Stability: Stable.
func NewWarmUpLR[T utils.Float](lr0 T, warmupSteps uint) *WarmUpLR[T] {
	return &WarmUpLR[T]{
		lr0:         lr0,
		warmupSteps: warmupSteps,
		gran:        PerStep,
	}
}

// Step advances the schedule and returns the current rate.
// During warm-up (t < warmupSteps): lr = lr₀ × (t / warmupSteps).
// After warm-up (t ≥ warmupSteps): lr = lr₀ (held constant, LRS-4).
func (w *WarmUpLR[T]) Step() T {
	w.step++
	if w.step >= w.warmupSteps {
		return w.lr0
	}
	return T(float64(w.lr0) * float64(w.step) / float64(w.warmupSteps))
}

// Reset restores the scheduler to step 0 (LRS-5).
func (w *WarmUpLR[T]) Reset() { w.step = 0 }

// Granularity reports the step-call frequency.
func (w *WarmUpLR[T]) Granularity() Granularity { return w.gran }

// warmupState is the JSON-serialisable snapshot of WarmUpLR.
type warmupState struct {
	LR0         float64 `json:"lr0"`
	WarmupSteps uint    `json:"warmup_steps"`
	Step        uint    `json:"step"`
	Gran        uint8   `json:"gran"`
}

// SaveState serialises all WarmUpLR state to JSON (LRS-6).
func (w *WarmUpLR[T]) SaveState() ([]byte, error) {
	return json.Marshal(warmupState{
		LR0:         float64(w.lr0),
		WarmupSteps: w.warmupSteps,
		Step:        w.step,
		Gran:        uint8(w.gran),
	})
}

// LoadState restores from a SaveState blob.
func (w *WarmUpLR[T]) LoadState(data []byte) error {
	var st warmupState
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	w.lr0 = T(st.LR0)
	w.warmupSteps = st.WarmupSteps
	w.step = st.Step
	w.gran = Granularity(st.Gran)
	return nil
}
