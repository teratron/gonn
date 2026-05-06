package main

import "testing"

// TestE07Compiles verifies the 2-hidden TanH + Linear(1) topology
// compiles and trains for one short loop without error.
func TestE07Compiles(t *testing.T) {
	if _, _, _, err := runE07(10); err != nil {
		t.Fatalf("runE07: %v", err)
	}
}

// TestE07RMSE asserts the spec target — test RMSE ≤ 0.10.
// Sine regression on a uniform grid is well-conditioned for a 2-hidden
// TanH network; Xavier init and 5000 epochs should clear this bar easily.
func TestE07RMSE(t *testing.T) {
	_, testRMSE, _, err := runE07(5000)
	if err != nil {
		t.Fatalf("runE07: %v", err)
	}
	if testRMSE > 0.10 {
		t.Errorf("test RMSE = %.4f, want ≤ 0.10", testRMSE)
	}
}
