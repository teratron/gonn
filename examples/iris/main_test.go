package main

import "testing"

// TestE05Compiles verifies the 2-hidden ReLU + SoftMax(3) topology
// compiles and trains for one short loop without error.
func TestE05Compiles(t *testing.T) {
	if _, _, _, err := runE05(7); err != nil {
		t.Fatalf("runE05: %v", err)
	}
}

// TestE05Accuracy asserts the spec target — train > 90 %, test > 85 %.
// Iris is well separated enough for a 2-hidden network to clear these
// thresholds comfortably after 5000 epochs of MSE + Xavier training.
func TestE05Accuracy(t *testing.T) {
	trainAcc, testAcc, _, err := runE05(2024)
	if err != nil {
		t.Fatalf("runE05: %v", err)
	}
	if trainAcc < 0.90 {
		t.Errorf("train accuracy = %.3f, want > 0.90", trainAcc)
	}
	if testAcc < 0.85 {
		t.Errorf("test accuracy = %.3f, want > 0.85", testAcc)
	}
}
