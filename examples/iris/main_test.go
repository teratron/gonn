package main

import "testing"

// TestCompiles verifies the 2-hidden ReLU + SoftMax(3) topology
// compiles and trains for one short loop without error.
func TestCompiles(t *testing.T) {
	if _, _, _, err := run(7); err != nil {
		t.Fatalf("run: %v", err)
	}
}

// TestAccuracy asserts the target — train > 90 %, test > 85 %. Iris is
// well separated enough for a 2-hidden network to clear these thresholds
// comfortably after 5000 epochs of MSE + Xavier training.
func TestAccuracy(t *testing.T) {
	trainAcc, testAcc, _, err := run(2024)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if trainAcc < 0.90 {
		t.Errorf("train accuracy = %.3f, want > 0.90", trainAcc)
	}
	if testAcc < 0.85 {
		t.Errorf("test accuracy = %.3f, want > 0.85", testAcc)
	}
}
