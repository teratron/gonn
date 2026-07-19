package nn

import (
	"errors"
	"sync/atomic"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// newCallbacksNet returns a minimal 2→4→1 XOR network compiled with the supplied
// extra options. MaxIterations defaults to 3 so unit tests finish fast.
func newCallbacksNet[T utils.Float](extra ...Option[T]) *NN[T] {
	base := []Option[T]{
		WithInput[T](2),
		WithHiddenLayer[T](4, activation.SIGMOID),
		WithOutput[T](1, activation.SIGMOID),
		WithLearningRate[T](0.3),
		WithLoss[T](loss.MSE),
		WithMaxIterations[T](3),
	}
	return MustNew(append(base, extra...)...)
}

// ── Unit tests for internal callback primitives ──────────────────────────────

func TestInvokeOneRecoversPanic(t *testing.T) {
	t.Parallel()
	panicking := func(_ CallbackContext[float32]) error {
		panic("deliberate test panic")
	}
	// invokeOne must not propagate the panic and must return nil (CB-5).
	if err := invokeOne(panicking, CallbackContext[float32]{}); err != nil {
		t.Errorf("invokeOne returned non-nil after panic: %v", err)
	}
}

func TestFireEventNilShortCircuit(t *testing.T) {
	t.Parallel()
	// nil slice → zero overhead, nil return (CB-3).
	if err := fireEvent(nil, CallbackContext[float32]{}); err != nil {
		t.Errorf("fireEvent(nil) returned %v; want nil", err)
	}
}

func TestFireEventOrder(t *testing.T) {
	t.Parallel()
	var order []int
	fns := []CallbackFn[float32]{
		func(_ CallbackContext[float32]) error { order = append(order, 0); return nil },
		func(_ CallbackContext[float32]) error { order = append(order, 1); return nil },
		func(_ CallbackContext[float32]) error { order = append(order, 2); return nil },
	}
	if err := fireEvent(fns, CallbackContext[float32]{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(order) != 3 || order[0] != 0 || order[1] != 1 || order[2] != 2 {
		t.Errorf("callback order = %v; want [0 1 2]", order)
	}
}

func TestFireEventStopsOnErrStopTraining(t *testing.T) {
	t.Parallel()
	var called int
	fns := []CallbackFn[float32]{
		func(_ CallbackContext[float32]) error { called++; return ErrStopTraining },
		func(_ CallbackContext[float32]) error { called++; return nil },
	}
	err := fireEvent(fns, CallbackContext[float32]{})
	if !errors.Is(err, ErrStopTraining) {
		t.Errorf("expected ErrStopTraining; got %v", err)
	}
	if called != 1 {
		t.Errorf("second callback was called after ErrStopTraining; called = %d", called)
	}
}

func TestFireEventWrappedErrStopTraining(t *testing.T) {
	t.Parallel()
	fns := []CallbackFn[float32]{
		func(_ CallbackContext[float32]) error {
			return errors.Join(errors.New("patience"), ErrStopTraining)
		},
	}
	err := fireEvent(fns, CallbackContext[float32]{})
	if !errors.Is(err, ErrStopTraining) {
		t.Errorf("wrapped ErrStopTraining not detected; got %v", err)
	}
}

func TestSnapshotFromWeights(t *testing.T) {
	t.Parallel()
	w := []float32{1, 2, 3}
	snap := snapshotFromWeights(w, 7)
	if snap == nil {
		t.Fatal("snapshotFromWeights returned nil")
	}
	if snap.Epoch != 7 {
		t.Errorf("Epoch = %d; want 7", snap.Epoch)
	}
	// Mutation of original must not affect snapshot (deep copy).
	w[0] = 99
	if snap.Weights[0] != 1 {
		t.Errorf("snapshot not a copy: snap.Weights[0] = %v after source mutation", snap.Weights[0])
	}
}

func TestSnapshotFromWeightsNil(t *testing.T) {
	t.Parallel()
	if got := snapshotFromWeights[float32](nil, 1); got != nil {
		t.Errorf("expected nil for nil weights; got %v", got)
	}
}

func TestStopReasonPtr(t *testing.T) {
	t.Parallel()
	p := func() *StopReason {
		val := StopCallback
		return &val
	}()
	if *p != StopCallback {
		t.Errorf("*p = %v; want StopCallback", *p)
	}
}

func TestFormatStopReasonError(t *testing.T) {
	t.Parallel()
	s := formatStopReasonError(StopLoopError)
	if s == "" {
		t.Error("formatStopReasonError returned empty string")
	}
}

func TestCallbackContextFrom(t *testing.T) {
	t.Parallel()
	snap := &Snapshot[float32]{Epoch: 3}
	ctx := callbackContextFrom(5, float32(0.1), float32(0.05), 3, snap)
	if ctx.Iteration != 5 || ctx.Loss != 0.1 || ctx.MinLoss != 0.05 || ctx.MinIter != 3 {
		t.Errorf("callbackContextFrom fields wrong: %+v", ctx)
	}
	if ctx.Snapshot != snap {
		t.Errorf("Snapshot pointer not propagated")
	}
}

// ── Integration tests via Fit ─────────────────────────────────────────────────

func TestCallbackRegistrationOrder(t *testing.T) {
	t.Parallel()
	var order []int
	n := newCallbacksNet(
		WithOnIterationEnd(func(_ CallbackContext[float32]) error {
			order = append(order, 0)
			return nil
		}),
		WithOnIterationEnd(func(_ CallbackContext[float32]) error {
			order = append(order, 1)
			return nil
		}),
	)
	if _, _, err := n.Fit(xorDataset[float32]()); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// Three epochs → each with two callbacks in registration order.
	for i := 0; i+1 < len(order); i += 2 {
		if order[i] != 0 || order[i+1] != 1 {
			t.Errorf("wrong callback order at epoch %d/%d: got %v", i/2+1, 3, order[i:i+2])
		}
	}
}

func TestOnImprovementFoundStopTraining(t *testing.T) {
	t.Parallel()
	// Stop immediately on the first improvement (epoch 1) — Fit should
	// return after one completed epoch, not three.
	var improvementCalls int
	n := newCallbacksNet(
		WithOnImprovementFound(func(ctx CallbackContext[float32]) error {
			improvementCalls++
			return ErrStopTraining
		}),
	)
	completedEpochs, _, err := n.Fit(xorDataset[float32]())
	if err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}
	if completedEpochs > 1 {
		t.Errorf("completedEpochs = %d; expected ≤ 1 (stopped at first improvement)", completedEpochs)
	}
	if improvementCalls == 0 {
		t.Error("OnImprovementFound was never called")
	}
}

func TestCallbackPanicRecovery(t *testing.T) {
	t.Parallel()
	// A panicking OnIterationEnd must not abort Fit — CB-5 recovery.
	var afterPanic int
	n := newCallbacksNet(
		WithOnIterationEnd(func(_ CallbackContext[float32]) error {
			panic("deliberate test panic in Fit")
		}),
		WithOnIterationEnd(func(_ CallbackContext[float32]) error {
			afterPanic++
			return nil
		}),
	)
	completedEpochs, _, err := n.Fit(xorDataset[float32]())
	if err != nil {
		t.Fatalf("Fit returned error after callback panic: %v", err)
	}
	if completedEpochs == 0 {
		t.Error("Fit completed zero epochs — panic was not recovered")
	}
	// The second callback (after the panicking one) must still fire.
	if afterPanic == 0 {
		t.Error("second OnIterationEnd was never called after sibling panic recovery")
	}
}

func TestOnTrainEndFiresOnNormalExit(t *testing.T) {
	t.Parallel()
	var fired int32
	var capturedReason *StopReason
	n := newCallbacksNet(
		WithOnTrainEnd(func(ctx CallbackContext[float32]) error {
			atomic.AddInt32(&fired, 1)
			capturedReason = ctx.StopReason
			return nil
		}),
	)
	completedEpochs, _, err := n.Fit(xorDataset[float32]())
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if atomic.LoadInt32(&fired) != 1 {
		t.Errorf("OnTrainEnd fired %d times; want 1", fired)
	}
	if capturedReason == nil {
		t.Fatal("OnTrainEnd: StopReason is nil")
	}
	if completedEpochs == 3 && *capturedReason != StopMaxIterations {
		t.Errorf("StopReason = %v; want StopMaxIterations after exhausting 3 epochs", *capturedReason)
	}
}

func TestOnTrainEndFiresOnStopCallback(t *testing.T) {
	t.Parallel()
	var endFired int32
	var endReason StopReason
	n := newCallbacksNet(
		// Stop at first improvement, leaving OnTrainEnd to observe StopCallback.
		WithOnImprovementFound(func(_ CallbackContext[float32]) error {
			return ErrStopTraining
		}),
		WithOnTrainEnd(func(ctx CallbackContext[float32]) error {
			atomic.AddInt32(&endFired, 1)
			if ctx.StopReason != nil {
				endReason = *ctx.StopReason
			}
			return nil
		}),
	)
	if _, _, err := n.Fit(xorDataset[float32]()); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if atomic.LoadInt32(&endFired) != 1 {
		t.Errorf("OnTrainEnd fired %d times; want 1", endFired)
	}
	if endReason != StopCallback {
		t.Errorf("StopReason = %v; want StopCallback", endReason)
	}
}

func TestOnTrainEndContextEpochAndLoss(t *testing.T) {
	t.Parallel()
	var endCtx CallbackContext[float32]
	n := newCallbacksNet(
		WithOnTrainEnd(func(ctx CallbackContext[float32]) error {
			endCtx = ctx
			return nil
		}),
	)
	completedEpochs, lastLoss, err := n.Fit(xorDataset[float32]())
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if endCtx.Iteration != int(completedEpochs) {
		t.Errorf("OnTrainEnd Iteration = %d; want %d", endCtx.Iteration, completedEpochs)
	}
	if endCtx.Loss != lastLoss {
		t.Errorf("OnTrainEnd Loss = %v; want %v", endCtx.Loss, lastLoss)
	}
}

// ── Benchmark ────────────────────────────────────────────────────────────────

// BenchmarkNoCallbacks verifies the zero-overhead guarantee: a Fit loop on a
// network with no registered callbacks must report 0 extra allocations vs. the
// baseline. Run with -benchmem to observe.
func BenchmarkNoCallbacks(b *testing.B) {
	data := xorDataset[float32]()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		nn := MustNew(
			WithInput[float32](2),
			WithHiddenLayer[float32](4, activation.SIGMOID),
			WithOutput[float32](1, activation.SIGMOID),
			WithLearningRate[float32](0.3),
			WithLoss[float32](loss.MSE),
			WithMaxIterations[float32](3),
		)
		_, _, _ = nn.Fit(data)
	}
}
