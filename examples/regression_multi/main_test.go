package main

import "testing"

// TestCompiles verifies the 2-hidden ReLU + Linear(3) topology compiles
// and trains for one short loop without error.
func TestCompiles(t *testing.T) {
	if _, _, _, err := run(7, 10); err != nil {
		t.Fatalf("run: %v", err)
	}
}

// TestRMSE asserts the target — per-dimension test RMSE ≤ 0.20. The
// three target functions are smooth and well-conditioned for a 2-hidden
// ReLU network with He init and 5000 training epochs.
func TestRMSE(t *testing.T) {
	_, testRMSE, _, err := run(42, 5000)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	const limit = 0.20
	for d, r := range testRMSE {
		if r > limit {
			t.Errorf("dim %d: test RMSE = %.4f, want ≤ %.2f", d, r, limit)
		}
	}
}
