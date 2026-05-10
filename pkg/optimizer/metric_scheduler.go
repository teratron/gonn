package optimizer

import "github.com/teratron/gonn/pkg/utils"

// MetricScheduler extends Scheduler[T] with a metric-aware stepping method.
// Implementations such as ReduceOnPlateau use the supplied metric (typically
// epoch loss) to decide whether to reduce the rate; implementations like
// OneCycleLR accept the call but ignore the metric (self-contained curve).
//
// The training loop type-asserts n.sched to MetricScheduler[T] after each
// epoch and calls StepWithMetric(epochLoss) when the assertion succeeds;
// otherwise it falls back to the plain Step().
//
// AI-Meta:
//   - Purpose: Extend Scheduler with metric-aware step for plateau-based and one-cycle LR strategies.
//   - Usage: ms, ok := sched.(optimizer.MetricScheduler[T]); if ok { ms.StepWithMetric(loss) }.
//   - Related: [Scheduler], [ReduceOnPlateau], [OneCycleLR], [BindScheduler].
//   - Stability: Stable.
type MetricScheduler[T utils.Float] interface {
	Scheduler[T]

	// StepWithMetric advances the schedule using the supplied metric and
	// returns the new effective learning rate. Called by the training loop
	// at PerEpoch granularity when the scheduler implements this interface.
	StepWithMetric(metric T) T
}

// compile-time assertions (C26) — both new metric scheduler types must satisfy
// the full MetricScheduler contract before any training loop can rely on them.
var (
	_ MetricScheduler[float32] = (*ReduceOnPlateau[float32])(nil)
	_ MetricScheduler[float32] = (*OneCycleLR[float32])(nil)
)
