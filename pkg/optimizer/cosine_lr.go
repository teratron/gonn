package optimizer

import (
	"encoding/json"
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// CosineAnnealingLR decays the learning rate from lr₀ to lrMin following a
// cosine curve over tMax steps. After tMax steps the rate is held at lrMin
// (LRS-4). Formula: lrMin + 0.5(lr₀ − lrMin)(1 + cos(πt / tMax)).
//
// AI-Meta:
//   - Purpose: Cosine-annealing LR scheduler — smooth decay from lr₀ to lrMin over tMax steps.
//   - Usage: sched := optimizer.NewCosineAnnealingLR[float32](0.1, 1e-5, 1000); wire via BindScheduler.
//   - Lifecycle: Step advances counter; holds lrMin after tMax; Reset restores step 0.
//   - Related: [Scheduler], [BindScheduler], [WarmUpLR], [ChainScheduler].
//   - Stability: Stable.
type CosineAnnealingLR[T utils.Float] struct {
	gran  Granularity
	lr0   T    // initial learning rate (lr₀)
	lrMin T    // minimum learning rate
	step  uint // global step counter
	tMax  uint // total annealing steps
}

// compile-time interface verification (C26).
var _ Scheduler[float32] = (*CosineAnnealingLR[float32])(nil)

// NewCosineAnnealingLR returns a CosineAnnealingLR scheduler.
// lr0 is the starting rate, lrMin is the floor, tMax is the cycle length.
// Granularity defaults to PerEpoch.
//
// AI-Meta:
//   - Purpose: Construct a cosine-annealing scheduler with start, floor, and cycle length.
//   - Usage: sched := optimizer.NewCosineAnnealingLR[float64](0.01, 1e-6, 100).
//   - Related: [CosineAnnealingLR], [BindScheduler].
//   - Stability: Stable.
func NewCosineAnnealingLR[T utils.Float](lr0, lrMin T, tMax uint) *CosineAnnealingLR[T] {
	return &CosineAnnealingLR[T]{
		lr0:   lr0,
		lrMin: lrMin,
		tMax:  tMax,
		gran:  PerEpoch,
	}
}

// Step advances the schedule and returns the current rate.
// During annealing (t ≤ tMax): lrMin + 0.5(lr₀ − lrMin)(1 + cos(πt / tMax)).
// After tMax: lrMin is returned unchanged (LRS-4).
func (c *CosineAnnealingLR[T]) Step() T {
	c.step++
	if c.step >= c.tMax {
		return c.lrMin
	}
	ratio := math.Cos(math.Pi * float64(c.step) / float64(c.tMax))
	return T(float64(c.lrMin) + 0.5*(float64(c.lr0)-float64(c.lrMin))*(1+ratio))
}

// Reset restores the scheduler to step 0 (LRS-5).
func (c *CosineAnnealingLR[T]) Reset() { c.step = 0 }

// Granularity reports the step-call frequency.
func (c *CosineAnnealingLR[T]) Granularity() Granularity { return c.gran }

// cosineState is the JSON-serialisable snapshot of CosineAnnealingLR.
type cosineState struct {
	LR0   float64 `json:"lr0"`
	LRMin float64 `json:"lr_min"`
	TMax  uint    `json:"t_max"`
	Step  uint    `json:"step"`
	Gran  uint8   `json:"gran"`
}

// SaveState serialises all CosineAnnealingLR state to JSON (LRS-6).
func (c *CosineAnnealingLR[T]) SaveState() ([]byte, error) {
	return json.Marshal(cosineState{
		LR0:   float64(c.lr0),
		LRMin: float64(c.lrMin),
		TMax:  c.tMax,
		Step:  c.step,
		Gran:  uint8(c.gran),
	})
}

// LoadState restores from a SaveState blob.
func (c *CosineAnnealingLR[T]) LoadState(data []byte) error {
	var st cosineState
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	c.lr0 = T(st.LR0)
	c.lrMin = T(st.LRMin)
	c.tMax = st.TMax
	c.step = st.Step
	c.gran = Granularity(st.Gran)
	return nil
}
