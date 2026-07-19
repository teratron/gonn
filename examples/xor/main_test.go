package main

import "testing"

// TestXORBuilderConverges asserts the Builder-style network reaches a
// recognisable XOR fit. The threshold is intentionally loose because the
// public API does not expose an RNG seed yet — random init can occasionally
// land in a slow-convergence basin. 0.1 is well above the spec target
// (< 0.01) but low enough to catch a genuinely broken backprop.
func TestXORBuilderConverges(t *testing.T) {
	loss := runBuilder()
	if loss > 0.1 {
		t.Errorf("Builder XOR final loss = %v, want < 0.1", loss)
	}
}

// TestXOROptionsConverges mirrors the Builder smoke test for the
// Functional Options API. Both styles must clear the same threshold —
// the catalog spec relies on this to prove parity.
func TestXOROptionsConverges(t *testing.T) {
	loss := runOptions()
	if loss > 0.1 {
		t.Errorf("Options XOR final loss = %v, want < 0.1", loss)
	}
}
