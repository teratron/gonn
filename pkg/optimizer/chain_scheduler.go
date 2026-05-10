package optimizer

import (
	"encoding/json"

	"github.com/teratron/gonn/pkg/utils"
)

// SchedulerSegment pairs a Scheduler with a step-count duration window.
// ChainScheduler applies each segment's scheduler for its duration, then
// advances to the next segment (LRS-3).
//
// AI-Meta:
//   - Purpose: One stage in a ChainScheduler — a sub-scheduler paired with its duration.
//   - Related: [ChainScheduler], [Scheduler].
//   - Stability: Stable.
type SchedulerSegment[T utils.Float] struct {
	Scheduler Scheduler[T]
	Duration  uint // number of Step calls this segment is active
}

// ChainScheduler runs a sequence of sub-schedulers, switching to the next
// segment when the current one has exhausted its Duration. After all
// segments are exhausted the last segment holds its final rate (LRS-4).
// Reset propagates to all sub-schedulers (LRS-5).
//
// AI-Meta:
//   - Purpose: Compose multiple schedulers sequentially, each active for a given duration.
//   - Usage: sched := optimizer.NewChainScheduler([]SchedulerSegment[float32]{{WarmUpLR(...), 1000}, {CosineAnnealingLR(...), 9000}}).
//   - Lifecycle: Step advances through segments; Reset restores all sub-schedulers to initial state.
//   - Related: [Scheduler], [SchedulerSegment], [BindScheduler].
//   - Stability: Stable.
type ChainScheduler[T utils.Float] struct {
	segments    []SchedulerSegment[T]
	globalStep  uint // total Step calls across all segments
	segStep     uint // steps consumed in the current active segment
	activeIdx   int  // index into segments
	lastRate    T    // last rate returned by Step
}

// compile-time interface verification (C26).
var _ Scheduler[float32] = (*ChainScheduler[float32])(nil)

// NewChainScheduler returns a ChainScheduler over the given segments.
// Segments are applied in order; the Granularity of the first segment is
// used as the chain's granularity (callers should use consistent granularities).
//
// AI-Meta:
//   - Purpose: Construct a sequential multi-scheduler chain from a list of segments.
//   - Usage: sched := optimizer.NewChainScheduler[float64](segs).
//   - Related: [ChainScheduler], [SchedulerSegment].
//   - Stability: Stable.
func NewChainScheduler[T utils.Float](segments []SchedulerSegment[T]) *ChainScheduler[T] {
	return &ChainScheduler[T]{segments: segments}
}

// Step advances the global counter, delegates to the active segment, and
// transitions to the next segment when the current duration expires.
// Returns the last rate when all segments are exhausted (LRS-4).
func (c *ChainScheduler[T]) Step() T {
	c.globalStep++

	if c.activeIdx >= len(c.segments) {
		// All segments exhausted — hold last rate (LRS-4).
		return c.lastRate
	}

	seg := &c.segments[c.activeIdx]
	rate := seg.Scheduler.Step()
	c.lastRate = rate
	c.segStep++

	if c.segStep >= seg.Duration {
		// Transition to next segment: reset it with the current rate as base.
		c.activeIdx++
		c.segStep = 0
		if c.activeIdx < len(c.segments) {
			c.segments[c.activeIdx].Scheduler.Reset()
		}
	}

	return rate
}

// Reset restores all sub-schedulers, counters, and active index to initial
// state so the entire chain can be replayed (LRS-5).
func (c *ChainScheduler[T]) Reset() {
	for i := range c.segments {
		c.segments[i].Scheduler.Reset()
	}
	c.globalStep = 0
	c.segStep = 0
	c.activeIdx = 0
	c.lastRate = 0
}

// Granularity returns the granularity of the first segment, or PerEpoch when
// the segments slice is empty.
func (c *ChainScheduler[T]) Granularity() Granularity {
	if len(c.segments) == 0 {
		return PerEpoch
	}
	return c.segments[0].Scheduler.Granularity()
}

// chainState is the JSON-serialisable snapshot of ChainScheduler metadata.
// Sub-scheduler states are stored inline as raw JSON blobs.
type chainState struct {
	GlobalStep uint              `json:"global_step"`
	SegStep    uint              `json:"seg_step"`
	ActiveIdx  int               `json:"active_idx"`
	LastRate   float64           `json:"last_rate"`
	SubStates  []json.RawMessage `json:"sub_states"`
}

// SaveState serialises all ChainScheduler state including sub-scheduler
// states to JSON (LRS-6).
func (c *ChainScheduler[T]) SaveState() ([]byte, error) {
	subStates := make([]json.RawMessage, len(c.segments))
	for i, seg := range c.segments {
		blob, err := seg.Scheduler.SaveState()
		if err != nil {
			return nil, err
		}
		subStates[i] = blob
	}
	return json.Marshal(chainState{
		GlobalStep: c.globalStep,
		SegStep:    c.segStep,
		ActiveIdx:  c.activeIdx,
		LastRate:   float64(c.lastRate),
		SubStates:  subStates,
	})
}

// LoadState restores from a SaveState blob. The order and types of sub-schedulers
// must match the original construction — sub-state blobs are loaded positionally.
func (c *ChainScheduler[T]) LoadState(data []byte) error {
	var st chainState
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	if len(st.SubStates) != len(c.segments) {
		return utils.Newf(utils.ErrIntegrity,
			"ChainScheduler.LoadState: segment count mismatch: got %d, want %d",
			len(st.SubStates), len(c.segments))
	}
	for i, blob := range st.SubStates {
		if err := c.segments[i].Scheduler.LoadState(blob); err != nil {
			return err
		}
	}
	c.globalStep = st.GlobalStep
	c.segStep = st.SegStep
	c.activeIdx = st.ActiveIdx
	c.lastRate = T(st.LastRate)
	return nil
}
