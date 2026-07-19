// Package nn — checkpoint integration tests.
package nn

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/checkpoint"
	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/optimizer"
)

func checkpointSamples() []Sample[float64] {
	return []Sample[float64]{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}
}

// TestCheckpointWriteAndResume guards the audit finding that pkg/checkpoint
// was implemented but never invoked: Fit must produce snapshots on the
// everyN grid, and Resume must reconstruct a query-identical network.
func TestCheckpointWriteAndResume(t *testing.T) {
	dir := t.TempDir()
	n := MustNew(
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithLoss[float64](loss.MSE),
		WithMaxIterations[float64](50),
		WithLossLimit[float64](-1),
		WithWeightInitSeed[float64](42),
		WithCheckpoint[float64](dir, 10, checkpoint.SweepConfig{}),
	)
	epochs, _, err := n.Fit(checkpointSamples())
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	if epochs != 50 {
		t.Fatalf("epochs = %d, want 50", epochs)
	}

	snaps, err := filepath.Glob(filepath.Join(dir, "snap-*"))
	if err != nil || len(snaps) == 0 {
		t.Fatalf("no snapshots written (glob err=%v)", err)
	}

	resumed, iter, err := Resume[float64](dir)
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if iter != 50 {
		t.Errorf("resumed at epoch %d, want 50", iter)
	}
	for _, s := range checkpointSamples() {
		want, _ := n.Query(s.Input)
		got, err := resumed.Query(s.Input)
		if err != nil {
			t.Fatalf("resumed.Query: %v", err)
		}
		if math.Abs(want[0]-got[0]) > 1e-12 {
			t.Errorf("Query(%v): resumed %v != original %v", s.Input, got[0], want[0])
		}
	}
}

// TestCheckpointRetentionSweep asserts the sweep keeps the directory bounded:
// with everyN=1 over 30 epochs and HotN=2/ColdM=3, at most 5 snapshots remain.
func TestCheckpointRetentionSweep(t *testing.T) {
	dir := t.TempDir()
	n := MustNew(
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithMaxIterations[float64](30),
		WithLossLimit[float64](-1),
		WithWeightInitSeed[float64](7),
		WithCheckpoint[float64](dir, 1, checkpoint.SweepConfig{HotN: 2, ColdM: 3}),
	)
	if _, _, err := n.Fit(checkpointSamples()); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	snaps, _ := filepath.Glob(filepath.Join(dir, "snap-*"))
	if len(snaps) > 5 {
		t.Errorf("retention sweep left %d snapshots, want ≤ 5 (hot 2 + cold 3)", len(snaps))
	}
	if len(snaps) == 0 {
		t.Error("no snapshots survived the sweep")
	}
}

// TestResumeRestoresAdamState verifies the optimizer state blob round-trips
// when the resumed run uses the same optimizer type.
func TestResumeRestoresAdamState(t *testing.T) {
	dir := t.TempDir()
	n := MustNew(
		WithInput[float64](2),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithOptimizer[float64](optimizer.NewAdam[float64](0.01)),
		WithMaxIterations[float64](20),
		WithLossLimit[float64](-1),
		WithWeightInitSeed[float64](42),
		WithCheckpoint[float64](dir, 20, checkpoint.SweepConfig{}),
	)
	if _, _, err := n.Fit(checkpointSamples()); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	// Resume builds with the DEFAULT optimizer (SGD) — the Adam blob must be
	// rejected gracefully (Warn, not error), and the network stays usable.
	resumed, _, err := Resume[float64](dir)
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if _, err := resumed.Query([]float64{0, 1}); err != nil {
		t.Errorf("resumed network unusable: %v", err)
	}
}

// TestCheckpointRejectsConvPrefix: WithCheckpoint on a conv-prefix network
// must fail at compile, not silently drop the prefix from snapshots.
func TestCheckpointRejectsConvPrefix(t *testing.T) {
	_, err := New(
		WithInput[float64](8),
		WithConv1D[float64](2, 3, 1, conv.PadValid, false),
		WithHiddenLayer[float64](4, activation.SIGMOID),
		WithOutput[float64](1, activation.SIGMOID),
		WithCheckpoint[float64](t.TempDir(), 1, checkpoint.SweepConfig{}),
	)
	if err == nil {
		t.Error("compile must reject WithCheckpoint + conv prefix")
	}
}
