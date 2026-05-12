package nn

import (
	"errors"
	"fmt"

	"github.com/teratron/gonn/pkg/utils"
)

// ErrStopTraining is the sentinel error a callback returns to request early
// training termination. The training loop enters the min-loss rollback path
// identical to loss-limit exit (CB-6). Wrap with %w when re-raising so
// errors.Is still works up the call stack.
//
// AI-Meta:
//   - Purpose: Sentinel returned by callbacks to trigger early stopping with best-weight rollback.
//   - Usage: return fmt.Errorf("patience exceeded: %w", nn.ErrStopTraining) inside a CallbackFn.
//   - Related: [CallbackFn], [CallbackRegistry], [ErrStopTraining].
//   - Stability: Stable.
var ErrStopTraining = errors.New("stop training")

// StopReason classifies why a Fit call ended. Passed in CallbackContext.StopReason
// for OnTrainEnd callbacks so they can branch on how training terminated.
//
// AI-Meta:
//   - Purpose: Enum classifying the reason Fit returned; available to OnTrainEnd callbacks.
//   - Usage: if ctx.StopReason != nil && *ctx.StopReason == nn.StopCallback { ... }.
//   - Related: [CallbackContext], [ErrStopTraining].
//   - Stability: Stable.
type StopReason int

const (
	// StopLossLimit fires when mean epoch loss drops below LossLimit.
	StopLossLimit StopReason = iota
	// StopMaxIterations fires when MaxIterations epochs complete without reaching LossLimit.
	StopMaxIterations
	// StopContextCancel fires when the context passed to Fit is cancelled.
	StopContextCancel
	// StopExternalStop fires when Stop() is called from another goroutine.
	StopExternalStop
	// StopCallback fires when a callback returns ErrStopTraining.
	StopCallback
	// StopLoopError fires when an internal training error (e.g. shape mismatch) halts Fit.
	StopLoopError
)

// Snapshot is a read-only copy of the network's trainable weights at the moment
// of the callback event. Passed inside CallbackContext for OnImprovementFound and
// OnIterationEnd callbacks; nil for OnTrainEnd if the network was Stopped.
//
// AI-Meta:
//   - Purpose: Immutable weight copy provided to callbacks for inspection without network access.
//   - Usage: Read ctx.Snapshot.Weights to inspect current weights inside a CallbackFn.
//   - Concurrency: ReadSafe; slice is a copy, not a reference to live weights.
//   - Related: [CallbackContext], [CallbackFn].
//   - Stability: Stable.
type Snapshot[T utils.Float] struct {
	// Weights is a flat copy of all trainable weights at the time of the event.
	Weights []T
	// Epoch is the epoch number that produced this snapshot.
	Epoch uint
}

// CallbackContext carries the read-only event payload delivered to every
// registered callback. All fields are value types or immutable copies (CB-4).
//
// AI-Meta:
//   - Purpose: Immutable event payload passed to each registered callback during Fit.
//   - Concurrency: ReadSafe; all fields are values or read-only copies.
//   - Related: [CallbackFn], [CallbackRegistry], [ErrStopTraining].
//   - Stability: Stable.
type CallbackContext[T utils.Float] struct {
	// Iteration is the current epoch number (1-based).
	Iteration int
	// Loss is the mean loss for the current epoch.
	Loss T
	// MinLoss is the best (lowest) mean loss observed so far.
	MinLoss T
	// MinIter is the epoch at which MinLoss was first recorded.
	MinIter int
	// StopReason is non-nil only in OnTrainEnd callbacks; classifies why Fit ended.
	StopReason *StopReason
	// Snapshot holds a read-only weight copy; nil in OnTrainEnd when network is stopped.
	Snapshot *Snapshot[T]
}

// CallbackFn is the function type all registered callbacks must satisfy.
// Returning ErrStopTraining (or a wrapped form) requests early termination.
// Returning any other non-nil error is treated the same as ErrStopTraining
// for the iteration-level dispatch (CB-6); use ErrStopTraining explicitly.
// Panics are recovered by invokeOne and training continues (CB-5).
//
// AI-Meta:
//   - Purpose: Typed callback signature for all Fit training events.
//   - Usage: func myCallback(ctx nn.CallbackContext[float32]) error { ...; return nil }.
//   - Related: [CallbackContext], [ErrStopTraining], [CallbackRegistry].
//   - Stability: Stable.
type CallbackFn[T utils.Float] func(ctx CallbackContext[T]) error

// CallbackRegistry holds per-event callback slices. Each slice is nil until
// the first callback is registered (lazy allocation satisfies CB-3: zero overhead
// when no callbacks are registered — fireEvent short-circuits on nil).
//
// AI-Meta:
//   - Purpose: Per-event container for registered callbacks; nil slices produce zero overhead (CB-3).
//   - Usage: Populated via WithOnIterationEnd / WithOnImprovementFound / WithOnTrainEnd options.
//   - Related: [CallbackFn], [ErrStopTraining], [fireEvent].
//   - Stability: Stable.
type CallbackRegistry[T utils.Float] struct {
	OnIterationEnd     []CallbackFn[T]
	OnImprovementFound []CallbackFn[T]
	OnTrainEnd         []CallbackFn[T]
}

// invokeOne calls fn(ctx) and recovers from any panic. On panic it emits a
// structured Warn log wrapping ErrCallbackPanic and returns nil so the
// training loop continues (CB-5). Returns the callback's error on normal return.
func invokeOne[T utils.Float](fn CallbackFn[T], ctx CallbackContext[T]) (err error) {
	defer func() {
		if r := recover(); r != nil {
			panicErr := utils.Newf(utils.ErrCallbackPanic,
				"callback panicked: %v", r)
			utils.Logger.Warn("training callback panicked — continuing",
				"error", panicErr)
			err = nil
		}
	}()
	return fn(ctx)
}

// fireEvent iterates fns in registration order (CB-7), invoking each via
// invokeOne. Returns immediately on the first ErrStopTraining (or wrapped
// form) — remaining callbacks in the slice are NOT called. Returns nil when
// all callbacks return nil or when fns is nil (CB-3 zero-overhead path).
func fireEvent[T utils.Float](fns []CallbackFn[T], ctx CallbackContext[T]) error {
	if fns == nil {
		return nil
	}
	for _, fn := range fns {
		if err := invokeOne(fn, ctx); err != nil {
			if errors.Is(err, ErrStopTraining) {
				return err
			}
		}
	}
	return nil
}

// fireOnTrainEnd builds a final CallbackContext with the provided stop reason
// and fires all OnTrainEnd callbacks. Called via defer at the top of Fit so it
// runs regardless of the return path — normal, error, or panic (CB-8).
//
// stopReason and epochs/loss are passed by pointer so the deferred call reads
// the final values at Fit return time, not the values at defer setup time.
func fireOnTrainEnd[T utils.Float](reg *CallbackRegistry[T], stopReason *StopReason, epochs *uint, loss *T) {
	if reg == nil {
		return
	}
	ctx := CallbackContext[T]{
		StopReason: stopReason,
	}
	if epochs != nil {
		ctx.Iteration = int(*epochs)
	}
	if loss != nil {
		ctx.Loss = *loss
		ctx.MinLoss = *loss
	}
	// Ignore errors from OnTrainEnd; training is already finishing.
	_ = fireEvent(reg.OnTrainEnd, ctx)
}

// stopReasonPtr is a convenience helper that allocates a StopReason on the
// heap and returns its pointer — avoids repetitive &localVar patterns.
func stopReasonPtr(r StopReason) *StopReason {
	return &r
}

// callbackContextFrom constructs a CallbackContext from the current training state.
func callbackContextFrom[T utils.Float](epoch int, loss, minLoss T, minIter int, snap *Snapshot[T]) CallbackContext[T] {
	return CallbackContext[T]{
		Iteration: epoch,
		Loss:      loss,
		MinLoss:   minLoss,
		MinIter:   minIter,
		Snapshot:  snap,
	}
}

// snapshotFromWeights copies the supplied flat weights slice into a Snapshot.
func snapshotFromWeights[T utils.Float](weights []T, epoch uint) *Snapshot[T] {
	if weights == nil {
		return nil
	}
	cp := make([]T, len(weights))
	copy(cp, weights)
	return &Snapshot[T]{Weights: cp, Epoch: epoch}
}

// formatStopReasonError returns an error string for diagnostic purposes.
func formatStopReasonError(r StopReason) string {
	return fmt.Sprintf("training stopped: reason=%d", int(r))
}
