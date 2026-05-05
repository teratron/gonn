package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/compute/cpu"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/persistence"
)

// roundTripPreservesQuery is the shared assertion used by single- and
// multi-hidden round-trip smoke tests below. Threshold uses
// cpu.ToleranceF32 (1e-5) per PERS-4.
func roundTripPreservesQuery(t *testing.T, tc trainConfig, label string) {
	t.Helper()
	original, err := train(tc)
	if err != nil {
		t.Fatalf("[%s] train: %v", label, err)
	}
	originalOutputs, err := queryAll(original)
	if err != nil {
		t.Fatalf("[%s] queryAll: %v", label, err)
	}

	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	wPath := filepath.Join(dir, "weights.json")

	cfg := buildConfigDoc(tc)
	if err := persistence.WriteConfig(cfgPath, cfg); err != nil {
		t.Fatalf("[%s] WriteConfig: %v", label, err)
	}
	if err := persistence.WriteWeights(wPath, cfg, extractWeights(original, tc)); err != nil {
		t.Fatalf("[%s] WriteWeights: %v", label, err)
	}

	_, loaded, err := persistence.ReadWeights[float32](cfgPath, wPath)
	if err != nil {
		t.Fatalf("[%s] ReadWeights: %v", label, err)
	}

	rebuilt, err := buildBlank(tc)
	if err != nil {
		t.Fatalf("[%s] buildBlank: %v", label, err)
	}
	if err := installWeights(rebuilt, tc, loaded); err != nil {
		t.Fatalf("[%s] installWeights: %v", label, err)
	}
	rebuiltOutputs, err := queryAll(rebuilt)
	if err != nil {
		t.Fatalf("[%s] queryAll(rebuilt): %v", label, err)
	}

	for i := range originalOutputs {
		for j := range originalOutputs[i] {
			d := absDiff(originalOutputs[i][j], rebuiltOutputs[i][j])
			if d > cpu.ToleranceF32 {
				t.Errorf("[%s] Query[%d][%d] drift = %v, want ≤ %v (PERS-4)",
					label, i, j, d, cpu.ToleranceF32)
			}
		}
	}
}

// TestRoundTripPreservesQuery exercises the same flow as run() with the
// canonical single-hidden XOR topology — the v0.1 regression baseline.
func TestRoundTripPreservesQuery(t *testing.T) {
	roundTripPreservesQuery(t, xorTopology(), "single-hidden XOR")
}

// TestRoundTripMultiHidden exercises Track C's multi-hidden seam:
// extract / install must walk every Hiddens[i] entry, with mixed bias
// settings across the chain. Convergence isn't asserted — Fit's tiny
// epoch budget mirrors the smoke style of the v0.1 example.
func TestRoundTripMultiHidden(t *testing.T) {
	tc := trainConfig{
		inputSize: 2,
		hidden: []hiddenSpec{
			{size: 5, act: activation.SIGMOID, bias: true},
			{size: 3, act: activation.SIGMOID, bias: false},
		},
		output:    outputSpec{size: 1, act: activation.SIGMOID, bias: true},
		loss:      loss.MSE,
		rate:      0.3,
		maxIters:  200,
		lossLimit: -1, // run the full short loop
	}
	roundTripPreservesQuery(t, tc, "two-hidden mixed-bias")
}

// TestRunFlowProducesArtifacts asserts the public run() path completes
// and cleans up its temp directory.
func TestRunFlowProducesArtifacts(t *testing.T) {
	tmp := t.TempDir()
	cwd, _ := os.Getwd()
	defer os.Chdir(cwd)
	if err := os.Chdir(tmp); err != nil {
		t.Fatal(err)
	}
	if err := run("smoke", xorTopology()); err != nil {
		t.Fatalf("run: %v", err)
	}
}
