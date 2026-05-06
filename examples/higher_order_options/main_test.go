package main

import "testing"

// TestE13SequentialCompiles verifies Sequential(2, 16, ReLU) compiles
// and trains one short loop without error.
func TestE13SequentialCompiles(t *testing.T) {
	if _, err := runSequential(7, 10); err != nil {
		t.Fatalf("runSequential: %v", err)
	}
}

// TestE13DeepNetworkCompiles verifies DeepNetwork(32, 3, ReLU) compiles
// and trains one short loop without error.
func TestE13DeepNetworkCompiles(t *testing.T) {
	if _, err := runDeepNetwork(7, 10); err != nil {
		t.Fatalf("runDeepNetwork: %v", err)
	}
}

// TestE13SequentialAccuracy asserts Sequential iris test accuracy > 85 %.
func TestE13SequentialAccuracy(t *testing.T) {
	acc, err := runSequential(2024, 5000)
	if err != nil {
		t.Fatalf("runSequential: %v", err)
	}
	if acc < 0.85 {
		t.Errorf("Sequential test acc = %.3f, want > 0.85", acc)
	}
}

// TestE13DeepNetworkAccuracy asserts DeepNetwork iris test accuracy > 85 %.
func TestE13DeepNetworkAccuracy(t *testing.T) {
	acc, err := runDeepNetwork(2024, 5000)
	if err != nil {
		t.Fatalf("runDeepNetwork: %v", err)
	}
	if acc < 0.85 {
		t.Errorf("DeepNetwork test acc = %.3f, want > 0.85", acc)
	}
}
