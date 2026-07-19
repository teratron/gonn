package main

import "testing"

// TestCompiles verifies the 2-hidden TanH + Linear(1) topology compiles
// and runs a short training loop without error.
func TestCompiles(t *testing.T) {
	if _, _, _, err := run(42, 10); err != nil {
		t.Fatalf("run: %v", err)
	}
}

// TestRMSE asserts the target — test RMSE ≤ 0.10. Adam optimizer
// converges quickly on this interpolation task; 5000 epochs with
// lr=0.001 clears the bar with margin.
func TestRMSE(t *testing.T) {
	if testing.Short() {
		t.Skip("skipped in -short mode")
	}
	_, testRMSE, _, err := run(42, 5000)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if testRMSE > 0.10 {
		t.Errorf("test RMSE = %.4f, want ≤ 0.10", testRMSE)
	}
}
