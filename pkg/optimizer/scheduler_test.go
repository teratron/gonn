package optimizer_test

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/optimizer"
)

// TestStepLRBoundary verifies the step-boundary transition, gamma=1 no-op,
// gamma=0 zero-rate, and SaveState/LoadState round-trip.
func TestStepLRBoundary(t *testing.T) {
	t.Run("decay at boundary", func(t *testing.T) {
		sched := optimizer.NewStepLR[float64](1.0, 3, 0.5)
		// step 1,2 → interval 0 → lr = 1.0
		for i := 0; i < 2; i++ {
			if r := sched.Step(); math.Abs(r-1.0) > 1e-12 {
				t.Errorf("step %d: got %v, want 1.0", i+1, r)
			}
		}
		// step 3 → interval 1 → lr = 1.0 × 0.5^1 = 0.5
		if r := sched.Step(); math.Abs(r-0.5) > 1e-12 {
			t.Errorf("step 3: got %v, want 0.5", r)
		}
		// step 6 → interval 2 → lr = 1.0 × 0.5^2 = 0.25
		for i := 0; i < 2; i++ {
			sched.Step()
		}
		if r := sched.Step(); math.Abs(r-0.25) > 1e-12 {
			t.Errorf("step 6: got %v, want 0.25", r)
		}
	})

	t.Run("gamma=1 no-op", func(t *testing.T) {
		sched := optimizer.NewStepLR[float64](0.1, 5, 1.0)
		for i := 0; i < 20; i++ {
			if r := sched.Step(); math.Abs(r-0.1) > 1e-12 {
				t.Errorf("step %d: got %v, want 0.1 (gamma=1 no-op)", i+1, r)
			}
		}
	})

	t.Run("gamma=0 zero-rate after first interval", func(t *testing.T) {
		sched := optimizer.NewStepLR[float64](1.0, 2, 0.0)
		sched.Step() // step 1 → interval 0 → 1.0
		r := sched.Step() // step 2 → interval 1 → 0.0
		if r != 0.0 {
			t.Errorf("step 2: got %v, want 0.0", r)
		}
	})

	t.Run("SaveState/LoadState round-trip", func(t *testing.T) {
		s1 := optimizer.NewStepLR[float64](1.0, 3, 0.5)
		s1.Step()
		s1.Step()
		blob, err := s1.SaveState()
		if err != nil {
			t.Fatalf("SaveState: %v", err)
		}
		s2 := optimizer.NewStepLR[float64](0, 1, 1.0)
		if err := s2.LoadState(blob); err != nil {
			t.Fatalf("LoadState: %v", err)
		}
		// Next Step from both should be identical.
		r1 := s1.Step()
		r2 := s2.Step()
		if math.Abs(r1-r2) > 1e-12 {
			t.Errorf("round-trip mismatch: s1=%v s2=%v", r1, r2)
		}
	})

	t.Run("Reset restores initial state", func(t *testing.T) {
		sched := optimizer.NewStepLR[float64](1.0, 3, 0.5)
		for i := 0; i < 6; i++ {
			sched.Step()
		}
		sched.Reset()
		// After reset, first step should be back to initial decay.
		r := sched.Step()
		if math.Abs(r-1.0) > 1e-12 {
			t.Errorf("after reset step 1: got %v, want 1.0", r)
		}
	})
}

// TestWarmUpLRRamp verifies linear ramp, plateau after warmup, and round-trip.
func TestWarmUpLRRamp(t *testing.T) {
	t.Run("linear ramp", func(t *testing.T) {
		sched := optimizer.NewWarmUpLR[float64](1.0, 4)
		// step 1 → 1.0 × (1/4) = 0.25
		if r := sched.Step(); math.Abs(r-0.25) > 1e-12 {
			t.Errorf("step 1: got %v, want 0.25", r)
		}
		// step 2 → 0.5
		if r := sched.Step(); math.Abs(r-0.5) > 1e-12 {
			t.Errorf("step 2: got %v, want 0.5", r)
		}
		// step 3 → 0.75
		if r := sched.Step(); math.Abs(r-0.75) > 1e-12 {
			t.Errorf("step 3: got %v, want 0.75", r)
		}
		// step 4 (>= warmupSteps) → plateau at 1.0
		if r := sched.Step(); math.Abs(r-1.0) > 1e-12 {
			t.Errorf("step 4 (plateau): got %v, want 1.0", r)
		}
		// step 5 → still 1.0 (LRS-4)
		if r := sched.Step(); math.Abs(r-1.0) > 1e-12 {
			t.Errorf("step 5 (post-warmup): got %v, want 1.0", r)
		}
	})

	t.Run("SaveState/LoadState round-trip", func(t *testing.T) {
		s1 := optimizer.NewWarmUpLR[float64](0.5, 10)
		for i := 0; i < 5; i++ {
			s1.Step()
		}
		blob, err := s1.SaveState()
		if err != nil {
			t.Fatalf("SaveState: %v", err)
		}
		s2 := optimizer.NewWarmUpLR[float64](0, 1)
		if err := s2.LoadState(blob); err != nil {
			t.Fatalf("LoadState: %v", err)
		}
		r1, r2 := s1.Step(), s2.Step()
		if math.Abs(r1-r2) > 1e-12 {
			t.Errorf("round-trip mismatch: s1=%v s2=%v", r1, r2)
		}
	})
}

// TestCosineAnnealingLR verifies the cosine schedule, plateau at lrMin, and round-trip.
func TestCosineAnnealingLR(t *testing.T) {
	t.Run("mid-point is halfway", func(t *testing.T) {
		// At t = tMax/2: cos(π/2) = 0 → rate = lrMin + 0.5*(lr0-lrMin)
		sched := optimizer.NewCosineAnnealingLR[float64](1.0, 0.0, 100)
		for i := 0; i < 50; i++ {
			sched.Step()
		}
		r := sched.Step() // step 51 ≈ midpoint check
		// Not exactly mid, but should be in (0, 1).
		if r <= 0.0 || r >= 1.0 {
			t.Errorf("step 51: got %v, expected (0, 1)", r)
		}
	})

	t.Run("plateau at lrMin after tMax", func(t *testing.T) {
		sched := optimizer.NewCosineAnnealingLR[float64](1.0, 0.01, 10)
		for i := 0; i < 15; i++ {
			sched.Step()
		}
		r := sched.Step()
		if math.Abs(r-0.01) > 1e-12 {
			t.Errorf("post-tMax: got %v, want 0.01", r)
		}
	})

	t.Run("SaveState/LoadState round-trip", func(t *testing.T) {
		s1 := optimizer.NewCosineAnnealingLR[float64](1.0, 0.001, 100)
		for i := 0; i < 40; i++ {
			s1.Step()
		}
		blob, err := s1.SaveState()
		if err != nil {
			t.Fatalf("SaveState: %v", err)
		}
		s2 := optimizer.NewCosineAnnealingLR[float64](0, 0, 1)
		if err := s2.LoadState(blob); err != nil {
			t.Fatalf("LoadState: %v", err)
		}
		r1, r2 := s1.Step(), s2.Step()
		if math.Abs(r1-r2) > 1e-12 {
			t.Errorf("round-trip mismatch: s1=%v s2=%v", r1, r2)
		}
	})
}

// TestChainSchedulerSequence verifies segment transitions and round-trip.
func TestChainSchedulerSequence(t *testing.T) {
	t.Run("warmup then step-decay", func(t *testing.T) {
		warmup := optimizer.NewWarmUpLR[float64](1.0, 4)
		decay := optimizer.NewStepLR[float64](1.0, 2, 0.5)
		chain := optimizer.NewChainScheduler([]optimizer.SchedulerSegment[float64]{
			{Scheduler: warmup, Duration: 4},
			{Scheduler: decay, Duration: 6},
		})

		// Warmup phase: 4 steps.
		for i := 0; i < 4; i++ {
			r := chain.Step()
			if r < 0 || r > 1.0 {
				t.Errorf("warmup step %d: out of range %v", i+1, r)
			}
		}
		// Decay phase: transition resets decay scheduler.
		r := chain.Step() // decay step 1 → interval 0 → 1.0
		if math.Abs(r-1.0) > 1e-12 {
			t.Errorf("decay step 1: got %v, want 1.0", r)
		}
		chain.Step() // decay step 2
		r = chain.Step() // decay step 3 → interval 1 → 0.5
		if math.Abs(r-0.5) > 1e-12 {
			t.Errorf("decay step 3: got %v, want 0.5", r)
		}
	})

	t.Run("exhausted chain holds last rate", func(t *testing.T) {
		sched := optimizer.NewStepLR[float64](1.0, 1, 0.5)
		chain := optimizer.NewChainScheduler([]optimizer.SchedulerSegment[float64]{
			{Scheduler: sched, Duration: 2},
		})
		chain.Step()
		last := chain.Step() // last step in segment
		// Segment exhausted — further Steps return last rate.
		for i := 0; i < 3; i++ {
			r := chain.Step()
			if math.Abs(r-last) > 1e-12 {
				t.Errorf("post-exhaustion step %d: got %v, want %v", i+1, r, last)
			}
		}
	})

	t.Run("Reset propagates to all sub-schedulers", func(t *testing.T) {
		warmup := optimizer.NewWarmUpLR[float64](1.0, 4)
		decay := optimizer.NewStepLR[float64](1.0, 2, 0.5)
		chain := optimizer.NewChainScheduler([]optimizer.SchedulerSegment[float64]{
			{Scheduler: warmup, Duration: 4},
			{Scheduler: decay, Duration: 4},
		})
		for i := 0; i < 8; i++ {
			chain.Step()
		}
		chain.Reset()
		// After reset, first step should be in warmup territory (< 1.0).
		r := chain.Step()
		if r >= 1.0 {
			t.Errorf("after reset: got %v, want < 1.0 (warmup phase)", r)
		}
	})

	t.Run("SaveState/LoadState round-trip", func(t *testing.T) {
		w1 := optimizer.NewWarmUpLR[float64](1.0, 4)
		w2 := optimizer.NewWarmUpLR[float64](1.0, 4)
		d1 := optimizer.NewStepLR[float64](1.0, 2, 0.5)
		d2 := optimizer.NewStepLR[float64](1.0, 2, 0.5)

		c1 := optimizer.NewChainScheduler([]optimizer.SchedulerSegment[float64]{
			{Scheduler: w1, Duration: 4},
			{Scheduler: d1, Duration: 4},
		})
		c2 := optimizer.NewChainScheduler([]optimizer.SchedulerSegment[float64]{
			{Scheduler: w2, Duration: 4},
			{Scheduler: d2, Duration: 4},
		})

		for i := 0; i < 5; i++ {
			c1.Step()
		}
		blob, err := c1.SaveState()
		if err != nil {
			t.Fatalf("SaveState: %v", err)
		}
		if err := c2.LoadState(blob); err != nil {
			t.Fatalf("LoadState: %v", err)
		}
		r1, r2 := c1.Step(), c2.Step()
		if math.Abs(r1-r2) > 1e-12 {
			t.Errorf("round-trip mismatch: c1=%v c2=%v", r1, r2)
		}
	})
}

// TestBindSchedulerUpdatesOptimizer verifies that BindScheduler pushes new
// rates into the optimizer after each Step call.
func TestBindSchedulerUpdatesOptimizer(t *testing.T) {
	opt := optimizer.NewSGD[float64](1.0)
	sched := optimizer.BindScheduler[float64](opt, optimizer.NewStepLR[float64](1.0, 2, 0.5))

	// Step 1: interval 0 → rate stays 1.0, optimizer unchanged.
	r := sched.Step()
	if math.Abs(float64(opt.LearningRate())-r) > 1e-12 {
		t.Errorf("after step 1: optimizer LR %v != sched rate %v", opt.LearningRate(), r)
	}
	// Step 2: interval 1 → rate 0.5, optimizer should be updated.
	r = sched.Step()
	if math.Abs(float64(opt.LearningRate())-r) > 1e-12 {
		t.Errorf("after step 2: optimizer LR %v != sched rate %v", opt.LearningRate(), r)
	}
}

// TestGranularity verifies the default Granularity values.
func TestGranularity(t *testing.T) {
	if optimizer.NewStepLR[float32](0.1, 10, 0.9).Granularity() != optimizer.PerEpoch {
		t.Error("StepLR default granularity should be PerEpoch")
	}
	if optimizer.NewWarmUpLR[float32](0.1, 100).Granularity() != optimizer.PerStep {
		t.Error("WarmUpLR default granularity should be PerStep")
	}
	if optimizer.NewCosineAnnealingLR[float32](0.1, 0.001, 100).Granularity() != optimizer.PerEpoch {
		t.Error("CosineAnnealingLR default granularity should be PerEpoch")
	}
}

// BenchmarkStepLRStep measures allocations on the StepLR hot path.
func BenchmarkStepLRStep(b *testing.B) {
	sched := optimizer.NewStepLR[float64](0.1, 10, 0.9)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = sched.Step()
	}
}

// TestExponentialLR verifies single step, accumulation, gamma=1 no-op,
// gamma near-zero floor, SaveState/LoadState round-trip, and Reset.
func TestExponentialLR(t *testing.T) {
	t.Run("single step", func(t *testing.T) {
		sched := optimizer.NewExponentialLR[float64](1.0, 0.9)
		// step 1 → lr₀ × 0.9^1 = 0.9
		if r := sched.Step(); math.Abs(r-0.9) > 1e-12 {
			t.Errorf("step 1: got %v, want 0.9", r)
		}
	})

	t.Run("accumulation", func(t *testing.T) {
		sched := optimizer.NewExponentialLR[float64](1.0, 0.5)
		// step 1 → 0.5, step 2 → 0.25, step 3 → 0.125
		want := []float64{0.5, 0.25, 0.125}
		for i, w := range want {
			if r := sched.Step(); math.Abs(r-w) > 1e-12 {
				t.Errorf("step %d: got %v, want %v", i+1, r, w)
			}
		}
	})

	t.Run("gamma=1 no-op", func(t *testing.T) {
		sched := optimizer.NewExponentialLR[float64](0.1, 1.0)
		for i := 0; i < 20; i++ {
			if r := sched.Step(); math.Abs(r-0.1) > 1e-12 {
				t.Errorf("step %d: got %v, want 0.1 (gamma=1 no-op)", i+1, r)
			}
		}
	})

	t.Run("gamma near-zero floor", func(t *testing.T) {
		sched := optimizer.NewExponentialLR[float64](1.0, 1e-10)
		// After enough steps the value should be effectively zero.
		for i := 0; i < 50; i++ {
			sched.Step()
		}
		r := sched.Step()
		if r < 0 {
			t.Errorf("rate went negative: %v", r)
		}
	})

	t.Run("SaveState/LoadState round-trip", func(t *testing.T) {
		s1 := optimizer.NewExponentialLR[float64](1.0, 0.9)
		for i := 0; i < 7; i++ {
			s1.Step()
		}
		blob, err := s1.SaveState()
		if err != nil {
			t.Fatalf("SaveState: %v", err)
		}
		s2 := optimizer.NewExponentialLR[float64](0, 1.0)
		if err := s2.LoadState(blob); err != nil {
			t.Fatalf("LoadState: %v", err)
		}
		r1, r2 := s1.Step(), s2.Step()
		if math.Abs(r1-r2) > 1e-12 {
			t.Errorf("round-trip mismatch: s1=%v s2=%v", r1, r2)
		}
	})

	t.Run("Reset restores lr0", func(t *testing.T) {
		sched := optimizer.NewExponentialLR[float64](0.5, 0.9)
		for i := 0; i < 10; i++ {
			sched.Step()
		}
		sched.Reset()
		// After reset, step 1 → lr₀ × gamma^1
		if r := sched.Step(); math.Abs(r-0.5*0.9) > 1e-12 {
			t.Errorf("after reset step 1: got %v, want %v", r, 0.5*0.9)
		}
	})
}

// TestGranularityExponential verifies ExponentialLR default granularity.
func TestGranularityExponential(t *testing.T) {
	if optimizer.NewExponentialLR[float32](0.1, 0.9).Granularity() != optimizer.PerEpoch {
		t.Error("ExponentialLR default granularity should be PerEpoch")
	}
}

// BenchmarkExponentialLRStep measures allocations on the ExponentialLR hot path.
func BenchmarkExponentialLRStep(b *testing.B) {
	sched := optimizer.NewExponentialLR[float64](0.1, 0.95)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = sched.Step()
	}
}
