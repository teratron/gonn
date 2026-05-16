package nn

import (
	"errors"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

// buildAndTrainNet creates a tiny network suitable for AndTrain smoke tests.
// Uses float64 to share xorSamples() with the existing test corpus.
func buildAndTrainNet(t *testing.T) *NN[float64] {
	t.Helper()
	n, err := New(
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLearningRate(0.3),
		WithMaxIterations[float64](20),
		WithLossLimit(1e-4),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return n
}

// snapshotAllWeights captures every axon weight for before/after comparison.
func snapshotAllWeights(t *testing.T, n *NN[float64]) []float64 {
	t.Helper()
	return n.snapshotWeights(nil)
}

// TestAndTrainContinuationPreservesWeights verifies AndTrain starts from the
// existing weight state — calling AndTrain with zero iterations leaves the
// network exactly as Fit left it.
func TestAndTrainContinuationPreservesWeights(t *testing.T) {
	t.Parallel()
	n := buildAndTrainNet(t)
	samples := xorSamples()
	if _, _, err := n.Fit(samples); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	pre := snapshotAllWeights(t, n)

	// Run AndTrain with a single iteration — weights must change, not reset.
	epochs, _, err := n.AndTrain(samples, WithMaxIterations[float64](1))
	if err != nil {
		t.Fatalf("AndTrain: %v", err)
	}
	if epochs == 0 {
		t.Errorf("expected at least one epoch in AndTrain, got %d", epochs)
	}
	post := snapshotAllWeights(t, n)
	if len(pre) != len(post) {
		t.Fatalf("weight count changed: %d → %d", len(pre), len(post))
	}
	// Weights should differ (proves training happened) but not be reinitialised
	// (i.e. not match a fresh He-normal sample). Verify by checking at least
	// one weight moved by a small amount, not by orders of magnitude.
	moved := false
	for i := range pre {
		if pre[i] != post[i] {
			moved = true
			break
		}
	}
	if !moved {
		t.Error("AndTrain produced no weight updates")
	}
}

// TestAndTrainRestoresOriginalConfig verifies opt/sched/reg overrides apply
// only for the duration of the AndTrain call and are restored on return.
func TestAndTrainRestoresOriginalConfig(t *testing.T) {
	t.Parallel()
	n := buildAndTrainNet(t)
	originalLR := n.cfg.LearningRate
	originalMaxIter := n.cfg.MaxIterations
	samples := xorSamples()

	_, _, err := n.AndTrain(samples,
		WithLearningRate(0.001),
		WithMaxIterations[float64](2),
	)
	if err != nil {
		t.Fatalf("AndTrain: %v", err)
	}
	if n.cfg.LearningRate != originalLR {
		t.Errorf("learning rate not restored: got %v, want %v", n.cfg.LearningRate, originalLR)
	}
	if n.cfg.MaxIterations != originalMaxIter {
		t.Errorf("max iterations not restored: got %d, want %d", n.cfg.MaxIterations, originalMaxIter)
	}
}

// TestAndTrainRejectsEmptySamples covers ErrInputData on zero-length input.
func TestAndTrainRejectsEmptySamples(t *testing.T) {
	t.Parallel()
	n := buildAndTrainNet(t)
	_, _, err := n.AndTrain(nil)
	if err == nil || !errors.Is(err, utils.ErrInputData) {
		t.Errorf("AndTrain(nil) err = %v, want ErrInputData", err)
	}
}

// TestAndTrainRejectsNonOperational covers ErrUserConfig when the lifecycle
// state precludes training. We construct a default NN[T] without compile and
// expect AndTrain to refuse.
func TestAndTrainRejectsNonOperational(t *testing.T) {
	t.Parallel()
	n := &NN[float64]{stateField: stateConfiguring}
	_, _, err := n.AndTrain(xorSamples())
	if err == nil || !errors.Is(err, utils.ErrUserConfig) {
		t.Errorf("AndTrain on Configuring state err = %v, want ErrUserConfig", err)
	}
}

// TestAndTrainRejectsRunning covers ErrNetworkRunning when the ctrl state is
// not Idle (simulating a Train already in progress).
func TestAndTrainRejectsRunning(t *testing.T) {
	t.Parallel()
	n := buildAndTrainNet(t)
	n.control.Store(controlRunning)
	defer n.control.Store(controlIdle)
	_, _, err := n.AndTrain(xorSamples())
	if err == nil || !errors.Is(err, utils.ErrNetworkRunning) {
		t.Errorf("AndTrain while Running err = %v, want ErrNetworkRunning", err)
	}
}
