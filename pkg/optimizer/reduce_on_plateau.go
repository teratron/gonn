package optimizer

import (
	"encoding/json"

	"github.com/teratron/gonn/pkg/utils"
)

// ReduceOnPlateau reduces the learning rate when a metric has stopped
// improving. The scheduler monitors the metric every epoch (Granularity =
// PerEpoch) and multiplies the current rate by factor after patience
// consecutive epochs without sufficient improvement. Once minLR is reached
// the rate is held there (LRS-4).
//
// Mode "min" (default): improvement means the metric decreased by more than
// threshold. Mode "max": improvement means the metric increased by more than
// threshold.
//
// AI-Meta:
//   - Purpose: Patience-based LR reduction — decay rate when metric stalls.
//   - Usage: sched := optimizer.NewReduceOnPlateau[float32](0.1); wire via BindScheduler.
//   - Lifecycle: StepWithMetric(epochLoss) called per epoch; Step() delegates with metric=0.
//   - Related: [MetricScheduler], [BindScheduler], [OneCycleLR].
//   - Stability: Stable.
type ReduceOnPlateau[T utils.Float] struct {
	lr0           T
	current       T
	best          T
	patienceCount uint
	factor        float64
	patience      uint
	threshold     float64
	minLR         T
	mode          string
}

// compile-time assertion (C26).
var _ MetricScheduler[float32] = (*ReduceOnPlateau[float32])(nil)

// ReduceOnPlateauOption is a functional option for [NewReduceOnPlateau].
//
// AI-Meta:
//   - Purpose: Functional option type for configuring ReduceOnPlateau.
//   - Related: [NewReduceOnPlateau], [WithROPFactor], [WithROPPatience].
//   - Stability: Stable.
type ReduceOnPlateauOption[T utils.Float] func(*ReduceOnPlateau[T])

// WithROPFactor sets the multiplicative reduction factor applied when the
// metric plateaus. Must satisfy 0 < factor < 1; default 0.1.
//
// AI-Meta:
//   - Purpose: Override ReduceOnPlateau multiplicative decay factor.
//   - Related: [NewReduceOnPlateau], [ReduceOnPlateauOption].
//   - Stability: Stable.
func WithROPFactor[T utils.Float](f float64) ReduceOnPlateauOption[T] {
	return func(r *ReduceOnPlateau[T]) { r.factor = f }
}

// WithROPPatience sets the number of consecutive epochs with no improvement
// that must elapse before the learning rate is reduced. Default 10.
//
// AI-Meta:
//   - Purpose: Override ReduceOnPlateau patience window.
//   - Related: [NewReduceOnPlateau], [ReduceOnPlateauOption].
//   - Stability: Stable.
func WithROPPatience[T utils.Float](p uint) ReduceOnPlateauOption[T] {
	return func(r *ReduceOnPlateau[T]) { r.patience = p }
}

// WithROPThreshold sets the minimum change in the monitored metric that
// qualifies as an improvement. Default 1e-4.
//
// AI-Meta:
//   - Purpose: Override ReduceOnPlateau improvement sensitivity.
//   - Related: [NewReduceOnPlateau], [ReduceOnPlateauOption].
//   - Stability: Stable.
func WithROPThreshold[T utils.Float](th float64) ReduceOnPlateauOption[T] {
	return func(r *ReduceOnPlateau[T]) { r.threshold = th }
}

// WithROPMinLR sets the lower bound on the learning rate. The scheduler
// will not reduce the rate below this value. Default 0.
//
// AI-Meta:
//   - Purpose: Set the minimum learning rate floor for ReduceOnPlateau.
//   - Related: [NewReduceOnPlateau], [ReduceOnPlateauOption].
//   - Stability: Stable.
func WithROPMinLR[T utils.Float](minLR T) ReduceOnPlateauOption[T] {
	return func(r *ReduceOnPlateau[T]) { r.minLR = minLR }
}

// WithROPMode sets the optimisation direction: "min" reduces on no decrease
// (suitable for loss), "max" reduces on no increase (suitable for accuracy).
// Default "min".
//
// AI-Meta:
//   - Purpose: Set ReduceOnPlateau monitoring direction ("min" or "max").
//   - Related: [NewReduceOnPlateau], [ReduceOnPlateauOption].
//   - Stability: Stable.
func WithROPMode[T utils.Float](mode string) ReduceOnPlateauOption[T] {
	return func(r *ReduceOnPlateau[T]) { r.mode = mode }
}

// NewReduceOnPlateau constructs a ReduceOnPlateau scheduler.
// lr0 is the initial learning rate; opts override defaults.
//
// AI-Meta:
//   - Purpose: Construct a patience-based plateau LR scheduler.
//   - Usage: sched := optimizer.NewReduceOnPlateau[float64](0.01, optimizer.WithROPPatience[float64](5)).
//   - Related: [ReduceOnPlateau], [BindScheduler].
//   - Stability: Stable.
func NewReduceOnPlateau[T utils.Float](lr0 T, opts ...ReduceOnPlateauOption[T]) *ReduceOnPlateau[T] {
	r := &ReduceOnPlateau[T]{
		lr0:       lr0,
		current:   lr0,
		factor:    0.1,
		patience:  10,
		threshold: 1e-4,
		mode:      "min",
	}
	for _, opt := range opts {
		opt(r)
	}
	if r.mode == "max" {
		r.best = -T(1e38)
	} else {
		r.best = T(1e38)
	}
	return r
}

// Step advances one epoch with a zero metric (delegates to StepWithMetric).
// Satisfies the plain Scheduler[T] interface; prefer StepWithMetric for
// metric-driven reductions.
func (r *ReduceOnPlateau[T]) Step() T { return r.StepWithMetric(0) }

// StepWithMetric advances the scheduler using the supplied metric value.
// If the metric does not improve for patience consecutive calls, the current
// learning rate is multiplied by factor (floored at minLR).
func (r *ReduceOnPlateau[T]) StepWithMetric(metric T) T {
	if r.isImproved(metric) {
		r.best = metric
		r.patienceCount = 0
	} else {
		r.patienceCount++
	}
	if r.patienceCount >= r.patience {
		r.patienceCount = 0
		candidate := T(float64(r.current) * r.factor)
		if candidate < r.minLR {
			candidate = r.minLR
		}
		r.current = candidate
	}
	return r.current
}

// isImproved returns true when metric shows sufficient improvement vs best.
func (r *ReduceOnPlateau[T]) isImproved(metric T) bool {
	if r.mode == "max" {
		return float64(metric) > float64(r.best)+r.threshold
	}
	return float64(metric) < float64(r.best)-r.threshold
}

// Reset restores the scheduler to its initial state (LRS-5).
func (r *ReduceOnPlateau[T]) Reset() {
	r.current = r.lr0
	r.patienceCount = 0
	if r.mode == "max" {
		r.best = -T(1e38)
	} else {
		r.best = T(1e38)
	}
}

// Granularity reports PerEpoch — ReduceOnPlateau always operates per epoch.
func (r *ReduceOnPlateau[T]) Granularity() Granularity { return PerEpoch }

// reducePlateauState is the JSON-serialisable snapshot of ReduceOnPlateau.
type reducePlateauState struct {
	LR0           float64 `json:"lr0"`
	Current       float64 `json:"current"`
	Best          float64 `json:"best"`
	PatienceCount uint    `json:"patience_count"`
	Factor        float64 `json:"factor"`
	Patience      uint    `json:"patience"`
	Threshold     float64 `json:"threshold"`
	MinLR         float64 `json:"min_lr"`
	Mode          string  `json:"mode"`
}

// SaveState serialises all ReduceOnPlateau state to JSON (LRS-6).
func (r *ReduceOnPlateau[T]) SaveState() ([]byte, error) {
	return json.Marshal(reducePlateauState{
		LR0:           float64(r.lr0),
		Current:       float64(r.current),
		Best:          float64(r.best),
		PatienceCount: r.patienceCount,
		Factor:        r.factor,
		Patience:      r.patience,
		Threshold:     r.threshold,
		MinLR:         float64(r.minLR),
		Mode:          r.mode,
	})
}

// LoadState restores from a SaveState blob.
func (r *ReduceOnPlateau[T]) LoadState(data []byte) error {
	var st reducePlateauState
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	r.lr0 = T(st.LR0)
	r.current = T(st.Current)
	r.best = T(st.Best)
	r.patienceCount = st.PatienceCount
	r.factor = st.Factor
	r.patience = st.Patience
	r.threshold = st.Threshold
	r.minLR = T(st.MinLR)
	r.mode = st.Mode
	return nil
}
