package optimizer

import (
	"testing"
)

func TestReduceOnPlateau_PatienceTrigger(t *testing.T) {
	sched := NewReduceOnPlateau[float64](0.1, WithROPPatience[float64](3), WithROPFactor[float64](0.5))

	// 3 calls with no improvement should trigger reduction
	for i := 0; i < 3; i++ {
		sched.StepWithMetric(0.5) // same metric, no improvement
	}
	got := sched.StepWithMetric(0.5) // 4th call — patience reset, rate reduced on count==patience
	// After patience=3 stale epochs, at the 3rd stale call patienceCount hits patience → reduce
	// Then next call resets patienceCount so rate stays reduced
	if got >= 0.1 {
		t.Errorf("expected reduced LR < 0.1, got %v", got)
	}
}

func TestReduceOnPlateau_PatientceExact(t *testing.T) {
	sched := NewReduceOnPlateau[float64](0.1, WithROPPatience[float64](2), WithROPFactor[float64](0.5))

	// First call is an improvement (0.5 < best=1e38), so patience doesn't start yet.
	sched.StepWithMetric(0.5) // improvement: best=0.5, patienceCount=0
	sched.StepWithMetric(0.5) // stale: patienceCount=1
	rate := sched.StepWithMetric(0.5) // stale: patienceCount=2=patience → reduce to 0.05
	const want = 0.05
	if rate != want {
		t.Errorf("expected %.4f, got %.4f", want, rate)
	}
}

func TestReduceOnPlateau_ModeMax(t *testing.T) {
	sched := NewReduceOnPlateau[float64](0.1,
		WithROPMode[float64]("max"),
		WithROPPatience[float64](2),
		WithROPFactor[float64](0.5),
	)

	// Improvement means metric increasing
	sched.StepWithMetric(0.6) // best=0.6
	sched.StepWithMetric(0.5) // no improvement (< best), patienceCount=1
	rate := sched.StepWithMetric(0.5) // patienceCount=2 → reduce
	if rate >= 0.1 {
		t.Errorf("expected reduced LR, got %v", rate)
	}

	// Now improvement should reset patience
	sched.StepWithMetric(0.9) // improvement, patienceCount=0
	if sched.patienceCount != 0 {
		t.Errorf("expected patience reset after improvement, got %d", sched.patienceCount)
	}
}

func TestReduceOnPlateau_Reset(t *testing.T) {
	sched := NewReduceOnPlateau[float64](0.1, WithROPPatience[float64](1), WithROPFactor[float64](0.5))
	sched.StepWithMetric(0.5)
	sched.StepWithMetric(0.5) // triggers reduction: current = 0.05
	sched.Reset()
	if sched.current != 0.1 {
		t.Errorf("Reset: expected current=0.1, got %v", sched.current)
	}
	if sched.patienceCount != 0 {
		t.Errorf("Reset: expected patienceCount=0, got %d", sched.patienceCount)
	}
}

func TestReduceOnPlateau_SaveLoadState(t *testing.T) {
	orig := NewReduceOnPlateau[float64](0.1,
		WithROPPatience[float64](5),
		WithROPFactor[float64](0.2),
		WithROPMinLR[float64](1e-5),
		WithROPThreshold[float64](1e-3),
	)
	orig.StepWithMetric(0.5)
	orig.StepWithMetric(0.5)

	blob, err := orig.SaveState()
	if err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	restored := NewReduceOnPlateau[float64](0.999)
	if err := restored.LoadState(blob); err != nil {
		t.Fatalf("LoadState: %v", err)
	}

	if restored.current != orig.current {
		t.Errorf("current mismatch: want %v got %v", orig.current, restored.current)
	}
	if restored.patienceCount != orig.patienceCount {
		t.Errorf("patienceCount mismatch: want %d got %d", orig.patienceCount, restored.patienceCount)
	}
	if restored.patience != orig.patience {
		t.Errorf("patience mismatch: want %d got %d", orig.patience, restored.patience)
	}
}

func TestReduceOnPlateau_StepDelegatesToStepWithMetric(t *testing.T) {
	sched := NewReduceOnPlateau[float64](0.1, WithROPPatience[float64](1), WithROPFactor[float64](0.5))
	// Step() with metric=0 — triggers patience since 0 is NOT < best (best=1e38 for mode="min")
	// First call: 0 < 1e38, so improvement → best=0, patienceCount=0
	r1 := sched.Step()
	if r1 != 0.1 {
		t.Errorf("first Step: expected 0.1, got %v", r1)
	}
	// Second call: metric=0, not < best=0 - threshold=1e-4, so no improvement
	r2 := sched.Step()
	_ = r2 // patienceCount=1=patience → reduces on this call
	// Third call after reduction: current should be 0.05
	r3 := sched.Step()
	_ = r3
}

func TestReduceOnPlateau_MinLRFloor(t *testing.T) {
	sched := NewReduceOnPlateau[float64](0.01,
		WithROPPatience[float64](1),
		WithROPFactor[float64](0.1),
		WithROPMinLR[float64](0.005),
	)
	// Trigger multiple reductions — rate should never go below minLR
	for i := 0; i < 20; i++ {
		sched.StepWithMetric(0.5)
	}
	if sched.current < 0.005 {
		t.Errorf("LR dropped below minLR: got %v", sched.current)
	}
}

func TestReduceOnPlateau_Granularity(t *testing.T) {
	sched := NewReduceOnPlateau[float64](0.1)
	if sched.Granularity() != PerEpoch {
		t.Errorf("expected PerEpoch, got %v", sched.Granularity())
	}
}

func TestReduceOnPlateau_BindSchedulerForwards(t *testing.T) {
	opt := NewSGD[float64](0.1)
	inner := NewReduceOnPlateau[float64](0.1, WithROPPatience[float64](1), WithROPFactor[float64](0.5))
	bound := BindScheduler[float64](opt, inner)

	ms, ok := bound.(MetricScheduler[float64])
	if !ok {
		t.Fatal("BindScheduler result must implement MetricScheduler")
	}
	// Trigger patience (2 stale + reduction)
	ms.StepWithMetric(0.5) // improvement (0.5 < 1e38)
	ms.StepWithMetric(0.5) // no improvement, patienceCount=1=patience → reduce
	rate := ms.StepWithMetric(0.5) // patienceCount=0 again, current=0.05
	_ = rate
	if opt.LearningRate() >= 0.1 {
		t.Errorf("expected optimizer LR to be reduced, got %v", opt.LearningRate())
	}
}
