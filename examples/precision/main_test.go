package main

import "testing"

// TestBothPrecisionsConverge asserts that the XOR network reaches a
// recognisable fit at both numeric precisions. Catches regressions in
// the generic plumbing — for example, a bug that only surfaces under
// float32 arithmetic would slip past the f64-only tests in pkg/nn.
func TestBothPrecisionsConverge(t *testing.T) {
	if l, _ := trainF32(); l > 0.15 {
		t.Errorf("f32 final loss = %v, want < 0.15", l)
	}
	if l, _ := trainF64(); l > 0.15 {
		t.Errorf("f64 final loss = %v, want < 0.15", l)
	}
}
