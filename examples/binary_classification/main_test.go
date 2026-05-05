package main

import "testing"

// TestE04Compiles is the cheap regression: the v0.2 multi-hidden BCE
// surface must Compile + Fit one short loop without error. Doesn't
// require accuracy — that's the next test's job.
func TestE04Compiles(t *testing.T) {
	if _, _, _, err := runE04(7); err != nil {
		t.Fatalf("runE04: %v", err)
	}
}

// TestE04Accuracy asserts the spec target — train > 90 %, test > 85 %
// — held loosely to absorb PCG seed drift across Go versions. Two-blob
// classification at σ = 0.6 with centres at ±1.5 is essentially linearly
// separable, so a 2-hidden ReLU network should clear these bars
// comfortably with the He init and 2000 epochs spec'd for E04.
func TestE04Accuracy(t *testing.T) {
	trainAcc, testAcc, _, err := runE04(2024)
	if err != nil {
		t.Fatalf("runE04: %v", err)
	}
	if trainAcc < 0.90 {
		t.Errorf("train accuracy = %.3f, want > 0.90", trainAcc)
	}
	if testAcc < 0.85 {
		t.Errorf("test accuracy = %.3f, want > 0.85", testAcc)
	}
}
