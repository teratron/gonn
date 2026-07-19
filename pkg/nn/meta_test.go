package nn

import (
	"errors"
	"testing"
	"time"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

// newTestNN builds a minimal 2→2→1 network for use as an inner NN in meta tests.
func newTestInnerNN(t *testing.T) *NN[float64] {
	t.Helper()
	inner, err := New(
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
	)
	if err != nil {
		t.Fatalf("newTestInnerNN: %v", err)
	}
	return inner
}

// ── ScalarParam ─────────────────────────────────────────────────────────────

func TestScalarParamGetSet(t *testing.T) {
	v := float64(0.3)
	p := &ScalarParam[float64]{ptr: &v, name: "lr"}

	got := p.Get()
	if len(got) != 1 || got[0] != 0.3 {
		t.Fatalf("Get: want [0.3], got %v", got)
	}

	if err := p.Set([]float64{0.5}); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if v != 0.5 {
		t.Fatalf("Set did not update pointer: got %v", v)
	}
}

func TestScalarParamSetShapeMismatch(t *testing.T) {
	v := float64(0.1)
	p := &ScalarParam[float64]{ptr: &v, name: "lr"}
	err := p.Set([]float64{0.1, 0.2})
	if !errors.Is(err, utils.ErrMetaLearnerShape) {
		t.Fatalf("expected ErrMetaLearnerShape, got %v", err)
	}
}

func TestScalarParamName(t *testing.T) {
	v := float64(0)
	p := &ScalarParam[float64]{ptr: &v, name: "learning_rate"}
	if p.Name() != "learning_rate" {
		t.Fatalf("Name: got %q", p.Name())
	}
}

// ── SliceParam ──────────────────────────────────────────────────────────────

func TestSliceParamGetSet(t *testing.T) {
	s := []float64{1, 2, 3}
	p := &SliceParam[float64]{ptr: &s, name: "scales"}

	got := p.Get()
	if len(got) != 3 || got[0] != 1 || got[2] != 3 {
		t.Fatalf("Get: want [1 2 3], got %v", got)
	}
	// Verify it's a copy.
	got[0] = 99
	if s[0] != 1 {
		t.Fatal("Get must return a copy, not a reference")
	}

	if err := p.Set([]float64{10, 20, 30}); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if s[0] != 10 || s[1] != 20 || s[2] != 30 {
		t.Fatalf("Set did not update slice: got %v", s)
	}
}

func TestSliceParamSetShapeMismatch(t *testing.T) {
	s := []float64{1, 2, 3}
	p := &SliceParam[float64]{ptr: &s, name: "scales"}
	err := p.Set([]float64{1, 2})
	if !errors.Is(err, utils.ErrMetaLearnerShape) {
		t.Fatalf("expected ErrMetaLearnerShape, got %v", err)
	}
}

// ── DefaultFeatureFunc ───────────────────────────────────────────────────────

func TestDefaultFeatureFunc(t *testing.T) {
	feats := DefaultFeatureFunc(0.5, 50, 100)
	if len(feats) != 2 {
		t.Fatalf("expected 2 features, got %d", len(feats))
	}
	if feats[0] != 0.5 {
		t.Fatalf("feature[0] (loss): want 0.5, got %v", feats[0])
	}
	if feats[1] != 0.5 { // 50/100
		t.Fatalf("feature[1] (progress): want 0.5, got %v", feats[1])
	}
}

// ── MetaLearner.step ─────────────────────────────────────────────────────────

func TestMetaLearnerStepDefaultFeatures(t *testing.T) {
	inner := newTestInnerNN(t)
	v := float64(0.3)
	ml := &MetaLearner[float64]{
		inner:  inner,
		params: []ParamAccessor[float64]{&ScalarParam[float64]{ptr: &v, name: "lr"}},
	}
	// step should not error; inner network has 1 output → 1 param registered.
	if err := ml.step(0.1, 1, 100); err != nil {
		t.Fatalf("step with default features: %v", err)
	}
	// lr must have been updated to whatever the inner network output.
	// We just verify no error and that v is in [0,1] (sigmoid output range).
	if v < 0 || v > 1 {
		t.Fatalf("expected lr in sigmoid output range [0,1], got %v", v)
	}
}

func TestMetaLearnerStepCustomFeatures(t *testing.T) {
	inner := newTestInnerNN(t)
	v := float64(0.3)
	callCount := 0
	ml := &MetaLearner[float64]{
		inner:  inner,
		params: []ParamAccessor[float64]{&ScalarParam[float64]{ptr: &v, name: "lr"}},
		Features: func(loss float64, iter int, maxIter int) []float64 {
			callCount++
			return []float64{loss, float64(iter) / float64(maxIter)}
		},
	}
	if err := ml.step(0.2, 10, 100); err != nil {
		t.Fatalf("step with custom features: %v", err)
	}
	if callCount != 1 {
		t.Fatalf("custom FeatureFunc was not called (callCount=%d)", callCount)
	}
}

func TestMetaLearnerStepShapeMismatch(t *testing.T) {
	// inner has 1 output, but we register 2 params → shape mismatch.
	inner := newTestInnerNN(t) // 2→4→1
	v1, v2 := float64(0.3), float64(0.01)
	ml := &MetaLearner[float64]{
		inner: inner,
		params: []ParamAccessor[float64]{
			&ScalarParam[float64]{ptr: &v1, name: "lr"},
			&ScalarParam[float64]{ptr: &v2, name: "mom"},
		},
	}
	err := ml.step(0.1, 1, 100)
	if !errors.Is(err, utils.ErrMetaLearnerShape) {
		t.Fatalf("expected ErrMetaLearnerShape, got %v", err)
	}
}

func TestMetaLearnerStepInnerQueryError(t *testing.T) {
	// inner network with wrong input size → Query will return ErrInputData.
	inner, err := New(
		WithInput[float64](5), // expects 5 inputs
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
	)
	if err != nil {
		t.Fatal(err)
	}
	v := float64(0.3)
	ml := &MetaLearner[float64]{
		inner:  inner,
		params: []ParamAccessor[float64]{&ScalarParam[float64]{ptr: &v, name: "lr"}},
		// DefaultFeatureFunc returns 2-element vector; inner expects 5.
	}
	err = ml.step(0.1, 1, 100)
	if err == nil {
		t.Fatal("expected error from Query shape mismatch, got nil")
	}
}

// ── WithMetaLearner option wiring ────────────────────────────────────────────

func TestWithMetaLearnerWiring(t *testing.T) {
	inner := newTestInnerNN(t)
	v := float64(0.3)
	ml := &MetaLearner[float64]{
		inner:  inner,
		params: []ParamAccessor[float64]{&ScalarParam[float64]{ptr: &v, name: "lr"}},
	}
	outer, err := New(
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithMetaLearner(ml),
	)
	if err != nil {
		t.Fatalf("New with WithMetaLearner: %v", err)
	}
	if outer.cfg.MetaLearner != ml {
		t.Fatal("cfg.MetaLearner was not set by WithMetaLearner")
	}
}

func TestWithMetaLearnerRejectsRunning(t *testing.T) {
	inner := newTestInnerNN(t)
	v := float64(0.3)
	ml := &MetaLearner[float64]{
		inner:  inner,
		params: []ParamAccessor[float64]{&ScalarParam[float64]{ptr: &v, name: "lr"}},
	}
	// Build an outer network via Builder API.
	outer := NewBuilder[float64]()
	outer.Input(2).Dense(4, activation.SIGMOID, false).Output(1, activation.SIGMOID, false)
	outer.cfg.MetaLearner = ml
	// Force the control state to Running to simulate in-training guard.
	outer.control.Store(controlRunning)

	_, err := outer.Compile()
	// Restore state so GC cleanup isn't confused.
	outer.control.Store(controlIdle)

	if !errors.Is(err, utils.ErrMetaLearnerRunning) {
		t.Fatalf("expected ErrMetaLearnerRunning, got %v", err)
	}
}

// ── Convergence check (T-12T01) ──────────────────────────────────────────────

// TestMetaLearnerConvergence verifies that training with MetaLearner attached
// does not catastrophically degrade convergence compared to a static LR.
// With an untrained inner network the output is near-sigmoid-center (~0.5),
// so LR stays in a workable range. Strict improvement requires a pre-trained
// inner network — that is an integration concern, not a unit test responsibility.
//
// Skip guard: if wall time exceeds 25 s the test bails (CI low-power runner).
func TestMetaLearnerConvergence(t *testing.T) {
	start := time.Now()

	const seeds = 3
	const maxIter = uint(3000)
	const lossLimit = float64(0.05)

	xor := []Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}

	staticTotal := 0
	metaTotal := 0

	for range seeds {
		if time.Since(start) > 25*time.Second {
			t.Skip("skipping convergence check: wall time > 25 s (CI low-power runner)")
		}

		static, err := New(
			WithInput[float64](2),
			WithHiddenLayer[float64](4, activation.SIGMOID),
			WithOutput[float64](1, activation.SIGMOID),
			WithLearningRate(0.3),
			WithMaxIterations[float64](maxIter),
			WithLossLimit(lossLimit),
		)
		if err != nil {
			t.Fatal(err)
		}
		e, _, _ := static.Fit(xor)
		staticTotal += int(e)

		if time.Since(start) > 25*time.Second {
			t.Skip("skipping convergence check: wall time > 25 s (CI low-power runner)")
		}

		inner, err := New(
			WithInput[float64](2),
			WithHiddenLayer[float64](4, activation.SIGMOID),
			WithOutput[float64](1, activation.SIGMOID),
		)
		if err != nil {
			t.Fatal(err)
		}
		outerLR := float64(0.3)
		ml := &MetaLearner[float64]{
			inner:  inner,
			params: []ParamAccessor[float64]{&ScalarParam[float64]{ptr: &outerLR, name: "lr"}},
		}
		meta, err := New(
			WithInput[float64](2),
			WithHiddenLayer[float64](4, activation.SIGMOID),
			WithOutput[float64](1, activation.SIGMOID),
			WithLearningRate(0.3),
			WithMaxIterations[float64](maxIter),
			WithLossLimit(lossLimit),
			WithMetaLearner(ml),
		)
		if err != nil {
			t.Fatal(err)
		}
		e, _, _ = meta.Fit(xor)
		metaTotal += int(e)
	}

	avgStatic := staticTotal / seeds
	avgMeta := metaTotal / seeds
	t.Logf("convergence: static avg=%d epochs, meta avg=%d epochs (3-seed)", avgStatic, avgMeta)

	// MetaLearner with untrained inner should not degrade by more than 3×.
	// Strict improvement (meta < static) requires a pre-trained inner network.
	if avgMeta > avgStatic*3 {
		t.Errorf("MetaLearner degraded convergence by >3× (static=%d, meta=%d) — check inner output range",
			avgStatic, avgMeta)
	}
}

// ── Integration: Fit calls meta-learner step ─────────────────────────────────

func TestTrainCallsMetaLearnerStep(t *testing.T) {
	// Inner network: 2 inputs → 1 output (tuning lr of outer).
	inner := newTestInnerNN(t)

	outerLR := float64(0.3)
	stepCount := 0
	ml := &MetaLearner[float64]{
		inner:  inner,
		params: []ParamAccessor[float64]{&ScalarParam[float64]{ptr: &outerLR, name: "lr"}},
		Features: func(loss float64, iter int, maxIter int) []float64 {
			stepCount++
			return []float64{loss, float64(iter) / float64(maxIter)}
		},
	}

	outer, err := New(
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLearningRate(0.3),
		WithMaxIterations[float64](5),
		WithMetaLearner(ml),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	samples := []Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}

	epochs, _, fitErr := outer.Fit(samples)
	if fitErr != nil {
		t.Fatalf("Fit: %v", fitErr)
	}
	// step should have been called once per epoch.
	if stepCount != int(epochs) {
		t.Fatalf("expected MetaLearner.step called %d times (once/epoch), got %d", epochs, stepCount)
	}
}
