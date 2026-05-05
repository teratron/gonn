package main

import "testing"

// TestBuildCompiles verifies the spec-canonical 4-hidden perceptron
// topology (3 → Sigmoid(5) → ReLU(10) → Sigmoid(5) → SoftMax(2))
// compiles cleanly under v0.2. Catches Compile() regressions on the
// multi-hidden seam without paying for the full training run.
func TestBuildCompiles(t *testing.T) {
	if _, err := build(); err != nil {
		t.Fatalf("build: %v", err)
	}
}

// TestTrainProducesQuery runs an abbreviated training loop (the build()
// configuration caps at 5000 epochs in trainPerceptron — full 100k is
// for the binary). The smoke test asserts only that:
//
//   - training completes without error;
//   - the post-train query has the expected output shape (lenOutput).
//
// Numeric proximity to the published reference is intentionally NOT
// asserted — random init plus the spec's loose convergence target make
// per-element drift dependent on PCG seed. Convergence sanity is the
// pkg/network XOR test's job; this example documents API usage.
func TestTrainProducesQuery(t *testing.T) {
	n, err := build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	pred, err := trainPerceptron(n)
	if err != nil {
		t.Fatalf("trainPerceptron: %v", err)
	}
	if len(pred) != lenOutput {
		t.Errorf("Query output len = %d, want %d", len(pred), lenOutput)
	}
}
