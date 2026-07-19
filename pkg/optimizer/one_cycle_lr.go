package optimizer

import (
	"encoding/json"
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// OneCycleLR implements a 3-phase learning rate schedule:
//
//  1. Warm-up: linear ramp from startLR (= maxLR/divFactor) to maxLR
//     over warmupSteps = pctStart × totalSteps.
//  2. Cosine decay: cosine anneal from maxLR to finalLR (= maxLR/finalDiv)
//     over the remaining totalSteps − warmupSteps steps.
//  3. Hold (LRS-4): finalLR is returned unchanged after totalSteps.
//
// The metric supplied to StepWithMetric is ignored — the curve is
// self-contained and not driven by observed loss.
//
// AI-Meta:
//   - Purpose: Self-contained one-cycle LR schedule — warm-up then cosine decay to finalLR.
//   - Usage: sched := optimizer.NewOneCycleLR[float32](0.1, 1000); wire via BindScheduler.
//   - Lifecycle: Step (or StepWithMetric) called per training step (Granularity = PerStep).
//   - Related: [MetricScheduler], [BindScheduler], [ReduceOnPlateau].
//   - Stability: Stable.
type OneCycleLR[T utils.Float] struct {
	current    T
	maxLR      T
	divFactor  float64
	finalDiv   float64
	pctStart   float64
	step       uint
	totalSteps uint
	gran       Granularity
}

// compile-time assertion (C26).
var _ MetricScheduler[float32] = (*OneCycleLR[float32])(nil)

// OneCycleLROption is a functional option for [NewOneCycleLR].
//
// AI-Meta:
//   - Purpose: Functional option type for configuring OneCycleLR.
//   - Related: [NewOneCycleLR], [WithPctStart], [WithFinalDiv], [WithDivFactor].
//   - Stability: Stable.
type OneCycleLROption[T utils.Float] func(*OneCycleLR[T])

// WithPctStart sets the fraction of totalSteps dedicated to the warm-up
// phase. Must be in (0, 1); default 0.3.
//
// AI-Meta:
//   - Purpose: Override the warm-up phase length as a fraction of totalSteps.
//   - Related: [NewOneCycleLR], [OneCycleLROption].
//   - Stability: Stable.
func WithPctStart[T utils.Float](p float64) OneCycleLROption[T] {
	return func(o *OneCycleLR[T]) { o.pctStart = p }
}

// WithFinalDiv sets the divisor that determines the final (minimum) learning
// rate: finalLR = maxLR / finalDiv. Default 1e4.
//
// AI-Meta:
//   - Purpose: Override the final LR divisor for OneCycleLR decay endpoint.
//   - Related: [NewOneCycleLR], [OneCycleLROption].
//   - Stability: Stable.
func WithFinalDiv[T utils.Float](d float64) OneCycleLROption[T] {
	return func(o *OneCycleLR[T]) { o.finalDiv = d }
}

// WithDivFactor sets the divisor that determines the initial (start) learning
// rate: startLR = maxLR / divFactor. Default 25.
//
// AI-Meta:
//   - Purpose: Override the start LR divisor for OneCycleLR warm-up phase.
//   - Related: [NewOneCycleLR], [OneCycleLROption].
//   - Stability: Stable.
func WithDivFactor[T utils.Float](d float64) OneCycleLROption[T] {
	return func(o *OneCycleLR[T]) { o.divFactor = d }
}

// NewOneCycleLR constructs an OneCycleLR scheduler.
// maxLR is the peak learning rate; totalSteps is the training-step count.
// Granularity defaults to PerStep.
//
// AI-Meta:
//   - Purpose: Construct a one-cycle LR scheduler with peak LR and total step count.
//   - Usage: sched := optimizer.NewOneCycleLR[float64](0.01, 5000, optimizer.WithPctStart[float64](0.25)).
//   - Related: [OneCycleLR], [BindScheduler].
//   - Stability: Stable.
func NewOneCycleLR[T utils.Float](maxLR T, totalSteps uint, opts ...OneCycleLROption[T]) *OneCycleLR[T] {
	o := &OneCycleLR[T]{
		maxLR:      maxLR,
		totalSteps: totalSteps,
		pctStart:   0.3,
		divFactor:  25,
		finalDiv:   1e4,
		gran:       PerStep,
	}
	for _, opt := range opts {
		opt(o)
	}
	o.current = T(float64(maxLR) / o.divFactor)
	return o
}

// warmupSteps returns the number of steps in the warm-up phase.
func (o *OneCycleLR[T]) warmupSteps() uint {
	return uint(o.pctStart * float64(o.totalSteps))
}

// Step advances the schedule by one step and returns the new rate.
// Delegates to StepWithMetric since OneCycleLR is metric-independent.
func (o *OneCycleLR[T]) Step() T { return o.StepWithMetric(0) }

// StepWithMetric advances the schedule ignoring the metric value.
// Phase 1 (warm-up): linear from startLR to maxLR.
// Phase 2 (cosine): cosine decay from maxLR to finalLR.
// After totalSteps: finalLR is held (LRS-4).
func (o *OneCycleLR[T]) StepWithMetric(_ T) T {
	if o.step >= o.totalSteps {
		return o.current
	}
	o.step++
	ws := o.warmupSteps()
	startLR := float64(o.maxLR) / o.divFactor
	finalLR := float64(o.maxLR) / o.finalDiv

	if o.step <= ws {
		// Phase 1: linear warm-up
		if ws == 0 {
			o.current = o.maxLR
		} else {
			progress := float64(o.step) / float64(ws)
			o.current = T(startLR + progress*(float64(o.maxLR)-startLR))
		}
	} else {
		// Phase 2: cosine decay from maxLR to finalLR
		decaySteps := o.totalSteps - ws
		progress := float64(o.step-ws) / float64(decaySteps)
		cosVal := (1 + math.Cos(math.Pi*progress)) / 2
		o.current = T(finalLR + cosVal*(float64(o.maxLR)-finalLR))
	}
	return o.current
}

// Reset restores the scheduler to step 0 (LRS-5).
func (o *OneCycleLR[T]) Reset() {
	o.step = 0
	o.current = T(float64(o.maxLR) / o.divFactor)
}

// Granularity reports PerStep — OneCycleLR advances on every training step.
func (o *OneCycleLR[T]) Granularity() Granularity { return o.gran }

// oneCycleLRState is the JSON-serialisable snapshot of OneCycleLR.
type oneCycleLRState struct {
	MaxLR      float64 `json:"max_lr"`
	Current    float64 `json:"current"`
	Step       uint    `json:"step"`
	TotalSteps uint    `json:"total_steps"`
	PctStart   float64 `json:"pct_start"`
	FinalDiv   float64 `json:"final_div"`
	DivFactor  float64 `json:"div_factor"`
	Gran       uint8   `json:"gran"`
}

// SaveState serialises all OneCycleLR state to JSON (LRS-6).
func (o *OneCycleLR[T]) SaveState() ([]byte, error) {
	return json.Marshal(oneCycleLRState{
		MaxLR:      float64(o.maxLR),
		Current:    float64(o.current),
		Step:       o.step,
		TotalSteps: o.totalSteps,
		PctStart:   o.pctStart,
		FinalDiv:   o.finalDiv,
		DivFactor:  o.divFactor,
		Gran:       uint8(o.gran),
	})
}

// LoadState restores from a SaveState blob.
func (o *OneCycleLR[T]) LoadState(data []byte) error {
	var st oneCycleLRState
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	o.maxLR = T(st.MaxLR)
	o.current = T(st.Current)
	o.step = st.Step
	o.totalSteps = st.TotalSteps
	o.pctStart = st.PctStart
	o.finalDiv = st.FinalDiv
	o.divFactor = st.DivFactor
	o.gran = Granularity(st.Gran)
	return nil
}
