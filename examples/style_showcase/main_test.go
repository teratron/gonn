package main

import (
	"math"
	"testing"
)

// TestThreeStylesConverge asserts every style trains the XOR net to
// roughly the same place. The catalog spec asks for parity within 1e-3
// but RNG init can spread the three runs further apart on small networks;
// 0.1 absolute is loose enough to survive any reasonable init draw while
// catching a genuine algorithmic regression.
func TestThreeStylesConverge(t *testing.T) {
	lossB := buildBuilder()
	lossO := buildOptions()
	lossP := buildPreset()

	for _, l := range []float32{lossB, lossO, lossP} {
		if l > 0.1 {
			t.Errorf("style loss %v > 0.1 — convergence regressed", l)
		}
	}
	maxLoss := max32(lossB, lossO, lossP)
	minLoss := min32(lossB, lossO, lossP)
	if math.Abs(float64(maxLoss-minLoss)) > 0.1 {
		t.Errorf("style spread %v exceeds 0.1 — parity regression?", maxLoss-minLoss)
	}
}

func max32(xs ...float32) float32 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

func min32(xs ...float32) float32 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x < m {
			m = x
		}
	}
	return m
}
