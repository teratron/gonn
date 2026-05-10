package optimizer

import (
	"math"
	"testing"
)

func TestOneCycleLR_PhaseAtWarmupBoundary(t *testing.T) {
	// warmupSteps = int(0.3 * 100) = 30
	sched := NewOneCycleLR[float64](0.1, 100)
	startLR := 0.1 / 25 // maxLR / divFactor = 0.004

	// After warmupSteps steps, LR should equal maxLR
	var rate float64
	for i := 0; i < 30; i++ {
		rate = sched.Step()
	}
	if math.Abs(rate-0.1) > 1e-9 {
		t.Errorf("at warmup boundary (step 30): want maxLR=0.1, got %v", rate)
	}
	_ = startLR
}

func TestOneCycleLR_PhaseAtDecayEnd(t *testing.T) {
	// After totalSteps, LR should equal finalLR = maxLR / finalDiv = 0.1/1e4 = 1e-5
	sched := NewOneCycleLR[float64](0.1, 100)
	finalLR := 0.1 / 1e4

	var rate float64
	for i := 0; i < 100; i++ {
		rate = sched.Step()
	}
	if math.Abs(rate-finalLR) > 1e-9 {
		t.Errorf("at decay end (step 100): want finalLR=%v, got %v", finalLR, rate)
	}
}

func TestOneCycleLR_LRS4HoldAfterTotalSteps(t *testing.T) {
	sched := NewOneCycleLR[float64](0.1, 50)
	for i := 0; i < 50; i++ {
		sched.Step()
	}
	r1 := sched.Step()
	r2 := sched.Step()
	if r1 != r2 {
		t.Errorf("LRS-4: expected hold after totalSteps, got %v then %v", r1, r2)
	}
	finalLR := 0.1 / 1e4
	if math.Abs(r1-finalLR) > 1e-9 {
		t.Errorf("LRS-4: expected finalLR=%v, got %v", finalLR, r1)
	}
}

func TestOneCycleLR_SaveLoadState(t *testing.T) {
	orig := NewOneCycleLR[float64](0.05, 200,
		WithPctStart[float64](0.2),
		WithDivFactor[float64](10),
		WithFinalDiv[float64](1000),
	)
	for i := 0; i < 50; i++ {
		orig.Step()
	}

	blob, err := orig.SaveState()
	if err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	restored := NewOneCycleLR[float64](0.999, 1)
	if err := restored.LoadState(blob); err != nil {
		t.Fatalf("LoadState: %v", err)
	}

	if restored.step != orig.step {
		t.Errorf("step mismatch: want %d got %d", orig.step, restored.step)
	}
	if math.Abs(float64(restored.current)-float64(orig.current)) > 1e-12 {
		t.Errorf("current mismatch: want %v got %v", orig.current, restored.current)
	}
	// Bit-identical output after restore
	r1 := orig.Step()
	r2 := restored.Step()
	if math.Abs(r1-r2) > 1e-12 {
		t.Errorf("post-restore Step output mismatch: %v vs %v", r1, r2)
	}
}

func TestOneCycleLR_GranularityDefault(t *testing.T) {
	sched := NewOneCycleLR[float64](0.1, 100)
	if sched.Granularity() != PerStep {
		t.Errorf("expected PerStep, got %v", sched.Granularity())
	}
}

func TestOneCycleLR_Reset(t *testing.T) {
	sched := NewOneCycleLR[float64](0.1, 100)
	for i := 0; i < 30; i++ {
		sched.Step()
	}
	sched.Reset()
	if sched.step != 0 {
		t.Errorf("Reset: expected step=0, got %d", sched.step)
	}
	startLR := 0.1 / 25
	if math.Abs(float64(sched.current)-startLR) > 1e-9 {
		t.Errorf("Reset: expected current=startLR=%v, got %v", startLR, sched.current)
	}
}

func TestOneCycleLR_WarmupLinear(t *testing.T) {
	sched := NewOneCycleLR[float64](1.0, 100, WithPctStart[float64](0.5))
	// warmupSteps = 50; startLR = 1.0/25 = 0.04
	// step 25 → progress = 25/50 = 0.5 → LR = 0.04 + 0.5*(1.0-0.04) = 0.04 + 0.48 = 0.52
	var r float64
	for i := 0; i < 25; i++ {
		r = sched.Step()
	}
	want := 0.04 + 0.5*(1.0-0.04)
	if math.Abs(r-want) > 1e-9 {
		t.Errorf("midpoint of warmup: want %v, got %v", want, r)
	}
}

func TestOneCycleLR_StepWithMetricIgnoresMetric(t *testing.T) {
	s1 := NewOneCycleLR[float64](0.1, 100)
	s2 := NewOneCycleLR[float64](0.1, 100)
	r1 := s1.Step()
	r2 := s2.StepWithMetric(999.0)
	if r1 != r2 {
		t.Errorf("StepWithMetric should produce same output as Step, got %v vs %v", r1, r2)
	}
}
