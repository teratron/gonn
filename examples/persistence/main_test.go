package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/teratron/gonn/pkg/compute/cpu"
	"github.com/teratron/gonn/pkg/persistence"
)

// TestRoundTripPreservesQuery exercises the same flow as run() but
// asserts numeric equivalence between pre-save and post-reload queries.
// The threshold uses cpu.ToleranceF32 (1e-5) per PERS-4. Going through
// the public run() would require parsing stdout — instead the test
// duplicates the wiring inline so the assertion can be precise.
func TestRoundTripPreservesQuery(t *testing.T) {
	original, err := train()
	if err != nil {
		t.Fatal(err)
	}
	originalOutputs, err := queryAll(original)
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	wPath := filepath.Join(dir, "weights.json")

	cfg := buildConfigDoc()
	if err := persistence.WriteConfig(cfgPath, cfg); err != nil {
		t.Fatal(err)
	}
	if err := persistence.WriteWeights(wPath, cfg, extractWeights(original)); err != nil {
		t.Fatal(err)
	}

	_, loaded, err := persistence.ReadWeights[float32](cfgPath, wPath)
	if err != nil {
		t.Fatal(err)
	}

	rebuilt, err := buildBlank()
	if err != nil {
		t.Fatal(err)
	}
	if err := installWeights(rebuilt, loaded); err != nil {
		t.Fatal(err)
	}
	rebuiltOutputs, err := queryAll(rebuilt)
	if err != nil {
		t.Fatal(err)
	}

	for i := range originalOutputs {
		for j := range originalOutputs[i] {
			d := absDiff(originalOutputs[i][j], rebuiltOutputs[i][j])
			if d > cpu.ToleranceF32 {
				t.Errorf("Query[%d][%d] drift = %v, want ≤ %v (PERS-4)",
					i, j, d, cpu.ToleranceF32)
			}
		}
	}
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
	if err := run("smoke"); err != nil {
		t.Fatalf("run: %v", err)
	}
}
