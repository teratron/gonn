package optimizer

import "github.com/teratron/gonn/pkg/utils"

// Granularity controls when the training loop calls Scheduler.Step.
//
// AI-Meta:
//   - Purpose: Enum for scheduler step frequency — per epoch or per training step.
//   - Usage: Returned by Scheduler.Granularity to inform the training loop call site.
//   - Related: [Scheduler], [PerEpoch], [PerStep].
//   - Stability: Stable.
type Granularity uint8

const (
	// PerEpoch instructs the training loop to call Scheduler.Step once after
	// all batches in an epoch are processed (most common for global schedules).
	PerEpoch Granularity = iota

	// PerStep instructs the training loop to call Scheduler.Step once after
	// every single batch/sample weight update (useful for warm-up phases).
	PerStep
)

// Scheduler adjusts an optimizer's effective learning rate at defined training
// intervals. A scheduler wraps an optimizer — it does not replace it. The
// optimizer's LearningRate() returns the most recently applied rate.
//
// AI-Meta:
//   - Purpose: Contract for composable learning-rate scheduling strategies.
//   - Usage: Bind to an optimizer via BindScheduler; pass to nn.WithScheduler.
//   - Lifecycle: Created before training; Step called per epoch or per step; Reset before restart.
//   - Concurrency: SingleGoroutine; Step must not be called concurrently.
//   - Related: [BindScheduler], [StepLR], [WarmUpLR], [CosineAnnealingLR], [ChainScheduler].
//   - Stability: Stable.
type Scheduler[T utils.Float] interface {
	// Step advances the schedule by one interval and returns the new effective
	// learning rate. If the schedule has expired (LRS-4), the last computed
	// rate is returned unchanged.
	Step() T

	// Reset restores the scheduler to its initial state — step 0 and the
	// original learning rate configured at construction (LRS-5).
	Reset()

	// Granularity reports whether the training loop should call Step after
	// each epoch (PerEpoch) or after each training sample/batch (PerStep).
	Granularity() Granularity

	// SaveState serialises all internal scheduler state to JSON so that
	// a training checkpoint can restore the exact schedule position (LRS-6).
	SaveState() ([]byte, error)

	// LoadState restores from a SaveState blob. Returns an error if the blob
	// is malformed or was produced by a different scheduler type.
	LoadState([]byte) error
}

// LearningRateSetter is an optional extension interface that concrete optimizers
// may implement to allow external mutation of the effective learning rate.
// Schedulers use it via BindScheduler to push rate updates after each Step.
//
// AI-Meta:
//   - Purpose: Optional optimizer capability for runtime learning-rate updates by schedulers.
//   - Usage: Implemented by all built-in optimizers; checked via type-assertion in BindScheduler.
//   - Related: [BindScheduler], [Scheduler], [Optimizer].
//   - Stability: Stable.
type LearningRateSetter[T utils.Float] interface {
	// SetLearningRate replaces the optimizer's current effective learning rate.
	// Called by a bound scheduler after each Step.
	SetLearningRate(T)
}

// BindScheduler wires sched to opt so that every sched.Step() call
// automatically pushes the new rate into opt via LearningRateSetter.
// The initial rate is read from opt.LearningRate() at bind time and stored
// inside the scheduler — no modification to opt occurs until the first Step.
//
// If opt does not implement LearningRateSetter, sched is returned unchanged
// (graceful degradation: the schedule advances but the optimizer rate stays fixed).
//
// AI-Meta:
//   - Purpose: Connect a scheduler to an optimizer so Step auto-updates the optimizer rate.
//   - Usage: sched := optimizer.BindScheduler(myOpt, optimizer.NewStepLR[float32](0.1, 10, 0.5)).
//   - Related: [Scheduler], [LearningRateSetter], [Optimizer].
//   - Stability: Stable.
func BindScheduler[T utils.Float](opt Optimizer[T], sched Scheduler[T]) Scheduler[T] {
	setter, ok := opt.(LearningRateSetter[T])
	if !ok {
		return sched
	}
	return &boundScheduler[T]{inner: sched, setter: setter}
}

// boundScheduler delegates all Scheduler methods to the wrapped inner scheduler
// and additionally calls setter.SetLearningRate after each Step.
type boundScheduler[T utils.Float] struct {
	inner  Scheduler[T]
	setter LearningRateSetter[T]
}

// compile-time assertion: boundScheduler forwards MetricScheduler so
// BindScheduler chains work transparently with ReduceOnPlateau / OneCycleLR.
var _ MetricScheduler[float32] = (*boundScheduler[float32])(nil)

func (b *boundScheduler[T]) Step() T {
	rate := b.inner.Step()
	b.setter.SetLearningRate(rate)
	return rate
}

// StepWithMetric forwards to the inner MetricScheduler if it implements the
// interface; otherwise falls back to the plain Step path.
func (b *boundScheduler[T]) StepWithMetric(metric T) T {
	var rate T
	if ms, ok := b.inner.(MetricScheduler[T]); ok {
		rate = ms.StepWithMetric(metric)
	} else {
		rate = b.inner.Step()
	}
	b.setter.SetLearningRate(rate)
	return rate
}

func (b *boundScheduler[T]) Reset()                      { b.inner.Reset() }
func (b *boundScheduler[T]) Granularity() Granularity    { return b.inner.Granularity() }
func (b *boundScheduler[T]) SaveState() ([]byte, error)  { return b.inner.SaveState() }
func (b *boundScheduler[T]) LoadState(data []byte) error { return b.inner.LoadState(data) }
