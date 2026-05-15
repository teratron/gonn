package optimizer_test

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/optimizer"
)

// TestSGDGolden verifies SGD applies the correct weight update.
func TestSGDGolden(t *testing.T) {
	opt := optimizer.NewSGD[float64](0.1)
	weights := []float64{1.0, 2.0}
	deltas := []float64{0.5, -1.0} // ∂L/∂w

	if err := opt.Step(weights, deltas); err != nil {
		t.Fatalf("Step error: %v", err)
	}
	// w[0] = 1.0 - 0.1*0.5  = 0.95
	// w[1] = 2.0 - 0.1*(-1) = 2.10
	want := []float64{0.95, 2.10}
	for i, got := range weights {
		if math.Abs(got-want[i]) > 1e-12 {
			t.Errorf("weights[%d]: got %v, want %v", i, got, want[i])
		}
	}
}

// TestZeroLRNoOp verifies all optimizer types return weights unchanged when lr=0.
func TestZeroLRNoOp(t *testing.T) {
	weights := []float64{1.0, -2.0, 3.0}
	deltas := []float64{0.1, 0.2, 0.3}
	original := make([]float64, len(weights))
	copy(original, weights)

	opts := []optimizer.Optimizer[float64]{
		optimizer.NewSGD[float64](0),
		optimizer.NewAdam[float64](0),
		optimizer.NewSGDMomentum[float64](0),
		optimizer.NewRMSProp[float64](0),
	}
	for _, opt := range opts {
		w := make([]float64, len(weights))
		copy(w, weights)
		if err := opt.Step(w, deltas); err != nil {
			t.Fatalf("Step error: %v", err)
		}
		for i, got := range w {
			if got != original[i] {
				t.Errorf("zero-lr modified weight[%d]: got %v, want %v", i, got, original[i])
			}
		}
	}
}

// TestAdamRoundTrip verifies SaveState → LoadState produces identical next Step.
func TestAdamRoundTrip(t *testing.T) {
	opt1 := optimizer.NewAdam[float64](0.01)
	opt2 := optimizer.NewAdam[float64](0.01)

	weights1 := []float64{1.0, 2.0, 3.0}
	weights2 := make([]float64, len(weights1))
	copy(weights2, weights1)
	deltas := []float64{0.1, -0.2, 0.05}

	// Warm up both with the same step.
	_ = opt1.Step(weights1, deltas)
	_ = opt2.Step(weights2, deltas)

	// Serialise opt1 state into opt3 and verify next Step is identical.
	blob, err := opt1.SaveState()
	if err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	opt3 := optimizer.NewAdam[float64](0)
	if err := opt3.LoadState(blob); err != nil {
		t.Fatalf("LoadState: %v", err)
	}

	// Apply second step.
	w1b := make([]float64, len(weights1))
	copy(w1b, weights1)
	w3b := make([]float64, len(weights1))
	copy(w3b, weights1)
	_ = opt2.Step(w1b, deltas)
	_ = opt3.Step(w3b, deltas)

	for i := range w1b {
		if math.Abs(w1b[i]-w3b[i]) > 1e-10 {
			t.Errorf("weights[%d] mismatch after round-trip: opt2=%v opt3=%v", i, w1b[i], w3b[i])
		}
	}
}

// TestResetIdempotent verifies Reset followed by identical Steps gives the same
// result as a fresh instance.
func TestResetIdempotent(t *testing.T) {
	opts := []struct {
		name string
		a    optimizer.Optimizer[float64]
		b    optimizer.Optimizer[float64]
	}{
		{"SGDMomentum", optimizer.NewSGDMomentum[float64](0.01), optimizer.NewSGDMomentum[float64](0.01)},
		{"Adam", optimizer.NewAdam[float64](0.01), optimizer.NewAdam[float64](0.01)},
		{"RMSProp", optimizer.NewRMSProp[float64](0.01), optimizer.NewRMSProp[float64](0.01)},
	}
	deltas := []float64{0.3, -0.1, 0.5}

	for _, tc := range opts {
		t.Run(tc.name, func(t *testing.T) {
			wa := []float64{1, 2, 3}
			wb := []float64{1, 2, 3}

			// Warm up a by 5 steps, then reset.
			for range 5 {
				_ = tc.a.Step(wa, deltas)
			}
			tc.a.Reset()
			// Reset to fresh weight values.
			copy(wa, []float64{1, 2, 3})

			// Both a (after reset) and b (fresh) should give identical first step.
			_ = tc.a.Step(wa, deltas)
			_ = tc.b.Step(wb, deltas)

			for i := range wa {
				if math.Abs(wa[i]-wb[i]) > 1e-10 {
					t.Errorf("[%s] weights[%d]: after-reset %v != fresh %v", tc.name, i, wa[i], wb[i])
				}
			}
		})
	}
}

// TestDefaultOptimizer verifies DefaultOptimizer returns a functioning SGD instance.
func TestDefaultOptimizer(t *testing.T) {
	opt := optimizer.DefaultOptimizer[float64](0.2)
	if opt.LearningRate() != 0.2 {
		t.Fatalf("LearningRate: got %v, want 0.2", opt.LearningRate())
	}
	weights := []float64{1.0}
	deltas := []float64{0.5}
	_ = opt.Step(weights, deltas)
	if math.Abs(weights[0]-0.9) > 1e-12 {
		t.Fatalf("Step: got %v, want 0.9", weights[0])
	}
}

// TestSGDInterface exercises Reset, LearningRate, SaveState, LoadState.
func TestSGDInterface(t *testing.T) {
	opt := optimizer.NewSGD[float64](0.5)

	if opt.LearningRate() != 0.5 {
		t.Fatalf("LearningRate: got %v, want 0.5", opt.LearningRate())
	}

	// Reset on stateless SGD is a no-op — must not panic.
	opt.Reset()

	blob, err := opt.SaveState()
	if err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	opt2 := optimizer.NewSGD[float64](0)
	if err := opt2.LoadState(blob); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if opt2.LearningRate() != 0.5 {
		t.Fatalf("LearningRate after LoadState: got %v, want 0.5", opt2.LearningRate())
	}

	// Invalid JSON must return error.
	if err := opt2.LoadState([]byte("not-json")); err == nil {
		t.Fatal("LoadState with bad JSON should return error")
	}
}

// TestAdamHyperAndLearningRate covers NewAdamHyper and LearningRate.
func TestAdamHyperAndLearningRate(t *testing.T) {
	opt := optimizer.NewAdamHyper[float64](0.001, 0.8, 0.99, 1e-7)
	if opt.LearningRate() != 0.001 {
		t.Fatalf("LearningRate: got %v, want 0.001", opt.LearningRate())
	}
	// Confirm Step runs without error.
	w := []float64{1.0, 2.0}
	d := []float64{0.1, -0.1}
	if err := opt.Step(w, d); err != nil {
		t.Fatalf("Step: %v", err)
	}
}

// TestRMSPropInterface covers NewRMSPropHyper, LearningRate, SaveState, LoadState.
func TestRMSPropInterface(t *testing.T) {
	opt := optimizer.NewRMSPropHyper[float64](0.01, 0.95, 1e-7)
	if opt.LearningRate() != 0.01 {
		t.Fatalf("LearningRate: got %v, want 0.01", opt.LearningRate())
	}

	w := []float64{1.0, -1.0}
	d := []float64{0.2, 0.3}
	_ = opt.Step(w, d) // prime internal buffer

	blob, err := opt.SaveState()
	if err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	opt2 := optimizer.NewRMSProp[float64](0)
	if err := opt2.LoadState(blob); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if opt2.LearningRate() != 0.01 {
		t.Fatalf("LearningRate after LoadState: got %v, want 0.01", opt2.LearningRate())
	}

	if err := opt2.LoadState([]byte("{bad")); err == nil {
		t.Fatal("LoadState with bad JSON should return error")
	}
}

// TestSGDMomentumInterface covers NewSGDMomentumWithGamma, LearningRate, SaveState, LoadState.
func TestSGDMomentumInterface(t *testing.T) {
	opt := optimizer.NewSGDMomentumWithGamma[float64](0.05, 0.95)
	if opt.LearningRate() != 0.05 {
		t.Fatalf("LearningRate: got %v, want 0.05", opt.LearningRate())
	}

	w := []float64{1.0, 2.0}
	d := []float64{0.1, -0.1}
	_ = opt.Step(w, d) // prime velocity buffer

	blob, err := opt.SaveState()
	if err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	opt2 := optimizer.NewSGDMomentum[float64](0)
	if err := opt2.LoadState(blob); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if opt2.LearningRate() != 0.05 {
		t.Fatalf("LearningRate after LoadState: got %v, want 0.05", opt2.LearningRate())
	}

	if err := opt2.LoadState([]byte("bad")); err == nil {
		t.Fatal("LoadState with bad JSON should return error")
	}
}

// BenchmarkSGDStep measures allocations on the SGD hot path — target 0 allocs/op.
func BenchmarkSGDStep(b *testing.B) {
	opt := optimizer.NewSGD[float64](0.01)
	weights := make([]float64, 256)
	deltas := make([]float64, 256)
	for i := range deltas {
		deltas[i] = float64(i) * 0.001
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = opt.Step(weights, deltas)
	}
}

// BenchmarkAdamStep measures allocations on Adam after warm-up — target 0 allocs/op.
func BenchmarkAdamStep(b *testing.B) {
	opt := optimizer.NewAdam[float64](0.001)
	weights := make([]float64, 256)
	deltas := make([]float64, 256)
	for i := range deltas {
		deltas[i] = float64(i) * 0.001
	}
	// Warm up to pre-allocate moment slices.
	_ = opt.Step(weights, deltas)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = opt.Step(weights, deltas)
	}
}
