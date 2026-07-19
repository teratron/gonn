package recurrent

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/utils"
)

func TestSimpleRNNForward(t *testing.T) {
	t.Parallel()
	cases := []struct {
		seqLen, inSize, hidden int
		seed                   uint64
	}{
		{2, 3, 4, 1001},
		{5, 4, 8, 1002},
		{3, 2, 2, 1003},
	}
	for _, tc := range cases {
		rng, _ := utils.NewRNG(tc.seed)
		r := NewSimpleRNN[float64](tc.seqLen, tc.inSize, tc.hidden)
		r.Init(rng)

		// Build deterministic input.
		input := make([]float64, tc.seqLen*tc.inSize)
		rng2, _ := utils.NewRNG(tc.seed + 1000)
		for i := range input {
			input[i] = rng2.Float64()*2 - 1
		}

		out := r.Forward(input)
		if len(out) != tc.seqLen*tc.hidden {
			t.Errorf("Forward output length = %d, want %d", len(out), tc.seqLen*tc.hidden)
		}
		for i, v := range out {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("Forward output[%d] = %v (non-finite)", i, v)
			}
			if math.Abs(v) > 1+1e-9 {
				t.Errorf("tanh output[%d] = %v, must be in (-1, 1)", i, v)
			}
		}

		// Deterministic: same input → same output.
		out2 := r.Forward(input)
		for i := range out {
			if out[i] != out2[i] {
				t.Errorf("Forward is not deterministic at index %d", i)
				break
			}
		}
	}
}

func TestSimpleRNNRoundTrip(t *testing.T) {
	t.Parallel()
	rng, _ := utils.NewRNG(42)
	r := NewSimpleRNN[float64](4, 3, 5)
	r.Init(rng)

	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}

	var r2 SimpleRNN[float64]
	if err := json.Unmarshal(data, &r2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}

	if r2.SeqLen != r.SeqLen || r2.InSize != r.InSize || r2.Hidden != r.Hidden {
		t.Errorf("shape mismatch after round-trip")
	}
	for i := range r.Wxh {
		if r.Wxh[i] != r2.Wxh[i] {
			t.Errorf("Wxh[%d] mismatch: got %v, want %v", i, r2.Wxh[i], r.Wxh[i])
			break
		}
	}
	for i := range r.Whh {
		if r.Whh[i] != r2.Whh[i] {
			t.Errorf("Whh[%d] mismatch: got %v, want %v", i, r2.Whh[i], r.Whh[i])
			break
		}
	}
}

// TestSimpleRNNGradient checks BPTT via finite differences (T-15T01).
func TestSimpleRNNGradient(t *testing.T) {
	t.Parallel()
	cases := []struct{ seqLen, inSize, hidden int }{
		{2, 3, 4},
		{5, 8, 4},
		{10, 1, 8},
	}
	const eps = 1e-5
	const tol = 1e-4

	for _, tc := range cases {
		rng, _ := utils.NewRNG(uint64(tc.seqLen*100 + tc.inSize*10 + tc.hidden))
		r := NewSimpleRNN[float64](tc.seqLen, tc.inSize, tc.hidden)
		r.Init(rng)

		rng2, _ := utils.NewRNG(99)
		input := make([]float64, tc.seqLen*tc.inSize)
		upstream := make([]float64, tc.seqLen*tc.hidden)
		for i := range input {
			input[i] = rng2.NormFloat64() * 0.5
		}
		for i := range upstream {
			upstream[i] = rng2.NormFloat64() * 0.5
		}

		// Analytical gradient via Backward.
		r.Forward(input)
		gradX := r.Backward(upstream)

		// Numerical gradient via finite differences.
		for j := range input {
			orig := input[j]

			input[j] = orig + eps
			outPlus := r.Forward(input)
			lPlus := dot(outPlus, upstream)

			input[j] = orig - eps
			outMinus := r.Forward(input)
			lMinus := dot(outMinus, upstream)

			input[j] = orig
			numerical := (lPlus - lMinus) / (2 * eps)
			analytical := gradX[j]
			rel := math.Abs(analytical-numerical) / (math.Abs(numerical) + 1e-8)
			if rel > tol {
				t.Errorf("seqLen=%d inSize=%d hidden=%d input[%d]: analytical=%.6g numerical=%.6g relErr=%.3e",
					tc.seqLen, tc.inSize, tc.hidden, j, analytical, numerical, rel)
			}
		}
	}
}

// dot computes the dot product of two equal-length slices.
func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
