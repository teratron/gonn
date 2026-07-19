package main

import "testing"

// TestCompiles is the cheap regression: the multi-hidden BCE surface
// must Compile + Fit one short loop without error. Doesn't require
// accuracy — that's the next test's job.
func TestCompiles(t *testing.T) {
	if _, _, _, err := run(7); err != nil {
		t.Fatalf("run: %v", err)
	}
}

// TestAccuracy asserts the target — train > 90 %, test > 85 % — held
// loosely to absorb PCG seed drift across Go versions. Two-blob
// classification at σ = 0.6 with centres at ±1.5 is essentially linearly
// separable, so a 2-hidden ReLU network should clear these bars
// comfortably with He init and 2000 epochs.
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
