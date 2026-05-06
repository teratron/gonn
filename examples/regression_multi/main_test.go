package main

import "testing"

// TestE08Compiles verifies the 2-hidden ReLU + Linear(3) topology
// compiles and trains for one short loop without error.
func TestE08Compiles(t *testing.T) {
	if _, _, _, err := runE08(7, 10); err != nil {
		t.Fatalf("runE08: %v", err)
	}
}

// TestE08RMSE asserts the spec target — per-dimension test RMSE ≤ 0.20.
// The three target functions are smooth and well-conditioned for a
// 2-hidden ReLU network with He init and 5000 training epochs.
func TestE08RMSE(t *testing.T) {
	_, testRMSE, _, err := runE08(42, 5000)
	if err != nil {
		t.Fatalf("runE08: %v", err)
	}
	const limit = 0.20
	for d, r := range testRMSE {
		if r > limit {
			t.Errorf("dim %d: test RMSE = %.4f, want ≤ %.2f", d, r, limit)
		}
	}
}
