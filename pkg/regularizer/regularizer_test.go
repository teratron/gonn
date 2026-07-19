package regularizer_test

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/regularizer"
)

// TestL2Penalty verifies L2 golden formula: λ=0.01, known weights.
func TestL2Penalty(t *testing.T) {
	reg := regularizer.NewL2[float64](0.01)
	weights := []float64{1.0, 2.0, 3.0}
	// expected = 0.01 × (1 + 4 + 9) = 0.14
	got := reg.Penalty(weights)
	want := 0.14
	if math.Abs(float64(got)-want) > 1e-12 {
		t.Errorf("L2.Penalty: got %v, want %v", got, want)
	}
}

// TestL1Penalty verifies L1 golden formula: λ=0.01, known weights.
func TestL1Penalty(t *testing.T) {
	reg := regularizer.NewL1[float64](0.01)
	weights := []float64{-1.0, 2.0, -3.0}
	// expected = 0.01 × (1 + 2 + 3) = 0.06
	got := reg.Penalty(weights)
	want := 0.06
	if math.Abs(float64(got)-want) > 1e-12 {
		t.Errorf("L1.Penalty: got %v, want %v", got, want)
	}
}

// TestDropoutInvertedScaling verifies retained acts are scaled by 1/p and
// the fraction of retained elements is approximately p.
func TestDropoutInvertedScaling(t *testing.T) {
	p := 0.8
	reg := regularizer.NewDropoutSeeded[float64](p, 42)
	n := 10000
	acts := make([]float64, n)
	for i := range acts {
		acts[i] = 1.0
	}
	out := reg.ApplyMask(acts, true)

	retained := 0
	for _, v := range out {
		if v != 0 {
			retained++
			// Inverted scaling: each retained value must be 1/p.
			want := 1.0 / p
			if math.Abs(v-want) > 1e-9 {
				t.Errorf("retained value not scaled: got %v, want %v", v, want)
			}
		}
	}
	// Fraction retained should be ≈ p with tolerance ±3%.
	frac := float64(retained) / float64(n)
	if math.Abs(frac-p) > 0.03 {
		t.Errorf("retention fraction %.3f too far from p=%.3f", frac, p)
	}
}

// TestInferenceDeterminism verifies ApplyMask(_, false) returns acts unchanged
// and consumes no RNG state.
func TestInferenceDeterminism(t *testing.T) {
	reg := regularizer.NewDropoutSeeded[float64](0.5, 1)
	acts := []float64{1.0, 2.0, 3.0}
	original := make([]float64, len(acts))
	copy(original, acts)

	out := reg.ApplyMask(acts, false)
	for i, v := range out {
		if v != original[i] {
			t.Errorf("inference path modified acts[%d]: got %v, want %v", i, v, original[i])
		}
	}
	// Second call should return the same values (no state consumed).
	out2 := reg.ApplyMask(acts, false)
	for i, v := range out2 {
		if v != original[i] {
			t.Errorf("inference path (2nd call) modified acts[%d]: got %v, want %v", i, v, original[i])
		}
	}
}

// TestComposeAdditivity verifies Compose(L2, Dropout).Penalty == L2.Penalty + Dropout.Penalty.
func TestComposeAdditivity(t *testing.T) {
	l2 := regularizer.NewL2[float64](0.01)
	drop := regularizer.NewDropoutSeeded[float64](0.8, 7)
	comp := regularizer.Compose[float64](l2, drop)

	weights := []float64{1.0, -2.0, 3.0}
	wantPenalty := l2.Penalty(weights) + drop.Penalty(weights)
	gotPenalty := comp.Penalty(weights)
	if math.Abs(float64(gotPenalty)-float64(wantPenalty)) > 1e-12 {
		t.Errorf("Compose.Penalty = %v, want %v", gotPenalty, wantPenalty)
	}
}

// TestDropoutBackwardMask verifies that BackwardMask zeros dropped positions and
// scales retained positions by 1/p, matching the forward mask.
func TestDropoutBackwardMask(t *testing.T) {
	p := 0.8
	reg := regularizer.NewDropoutSeeded[float64](p, 42)
	n := 1000
	acts := make([]float64, n)
	for i := range acts {
		acts[i] = 1.0
	}

	// Forward: apply mask, recording which elements are retained (non-zero).
	out := reg.ApplyMask(acts, true)
	retained := make([]bool, n)
	for i, v := range out {
		retained[i] = v != 0
	}

	// Backward: BackwardMask should apply the same mask to upstream.
	upstream := make([]float64, n)
	for i := range upstream {
		upstream[i] = 1.0
	}
	grad := reg.BackwardMask(upstream)

	scale := 1.0 / p
	for i := range grad {
		if retained[i] {
			if math.Abs(grad[i]-scale) > 1e-9 {
				t.Errorf("BackwardMask[%d]: retained, got %v want %v", i, grad[i], scale)
			}
		} else {
			if grad[i] != 0 {
				t.Errorf("BackwardMask[%d]: dropped, got %v want 0", i, grad[i])
			}
		}
	}
}

// TestDropoutBackwardMaskInference verifies BackwardMask returns upstream
// unchanged when the last ApplyMask call used training=false.
func TestDropoutBackwardMaskInference(t *testing.T) {
	reg := regularizer.NewDropoutSeeded[float64](0.5, 1)
	acts := []float64{1.0, 2.0, 3.0}
	reg.ApplyMask(acts, false) // inference mode: no mask stored

	upstream := []float64{0.1, 0.2, 0.3}
	grad := reg.BackwardMask(upstream)
	for i, v := range grad {
		if v != upstream[i] {
			t.Errorf("BackwardMask inference [%d]: got %v want %v", i, v, upstream[i])
		}
	}
}

// TestNilGuards verifies Apply and Penalty helpers are nil-safe.
func TestNilGuards(t *testing.T) {
	acts := []float64{1.0, 2.0, 3.0}
	out := regularizer.Apply[float64](nil, acts, true)
	for i, v := range out {
		if v != acts[i] {
			t.Errorf("Apply(nil) modified acts[%d]", i)
		}
	}
	p := regularizer.Penalty[float64](nil, acts)
	if p != 0 {
		t.Errorf("Penalty(nil) = %v, want 0", p)
	}
}
