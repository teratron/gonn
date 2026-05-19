package recurrent

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/utils"
)

func TestGRUForward(t *testing.T) {
	t.Parallel()
	cases := []struct {
		seqLen, inSize, hidden int
		seed                   uint64
	}{
		{2, 3, 4, 3001},
		{5, 4, 8, 3002},
		{3, 2, 2, 3003},
	}
	for _, tc := range cases {
		rng, _ := utils.NewRNG(tc.seed)
		g := NewGRU[float64](tc.seqLen, tc.inSize, tc.hidden)
		g.Init(rng)

		rng2, _ := utils.NewRNG(tc.seed + 1000)
		input := make([]float64, tc.seqLen*tc.inSize)
		for i := range input {
			input[i] = rng2.Float64()*2 - 1
		}

		out := g.Forward(input)
		if len(out) != tc.seqLen*tc.hidden {
			t.Errorf("Forward output length = %d, want %d", len(out), tc.seqLen*tc.hidden)
		}
		for i, v := range out {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("Forward output[%d] = %v (non-finite)", i, v)
			}
			// GRU h_t is a convex combination of hPrev and n_t (both bounded by tanh).
			// |h_t| < 1 holds when |h_0| = 0.
			if math.Abs(v) >= 1+1e-9 {
				t.Errorf("GRU output[%d] = %v, expected |h| < 1", i, v)
			}
		}

		// Deterministic.
		out2 := g.Forward(input)
		for i := range out {
			if out[i] != out2[i] {
				t.Errorf("Forward is not deterministic at index %d", i)
				break
			}
		}
	}
}

// TestGRUGradient checks BPTT via finite differences (T-16T01).
func TestGRUGradient(t *testing.T) {
	t.Parallel()
	cases := []struct{ seqLen, inSize, hidden int }{
		{1, 3, 4},
		{3, 4, 4},
		{5, 2, 6},
		{4, 3, 3}, // mismatched H_in / H_out covered by hidden != inSize
	}
	const eps = 1e-5
	const tol = 1e-4

	for _, tc := range cases {
		rng, _ := utils.NewRNG(uint64(tc.seqLen*100 + tc.inSize*10 + tc.hidden))
		g := NewGRU[float64](tc.seqLen, tc.inSize, tc.hidden)
		g.Init(rng)

		rng2, _ := utils.NewRNG(77)
		input := make([]float64, tc.seqLen*tc.inSize)
		upstream := make([]float64, tc.seqLen*tc.hidden)
		for i := range input {
			input[i] = rng2.NormFloat64() * 0.5
		}
		for i := range upstream {
			upstream[i] = rng2.NormFloat64() * 0.5
		}

		g.Forward(input)
		gradX := g.Backward(upstream)

		for j := range input {
			orig := input[j]

			input[j] = orig + eps
			outPlus := g.Forward(input)
			lPlus := dot(outPlus, upstream)

			input[j] = orig - eps
			outMinus := g.Forward(input)
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

func TestGRURoundTrip(t *testing.T) {
	t.Parallel()
	rng, _ := utils.NewRNG(55)
	g := NewGRU[float64](4, 3, 5)
	g.Init(rng)

	data, err := json.Marshal(g)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	var g2 GRU[float64]
	if err := json.Unmarshal(data, &g2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if g2.SeqLen != g.SeqLen || g2.InSize != g.InSize || g2.Hidden != g.Hidden {
		t.Errorf("shape mismatch after round-trip")
	}
	for i := range g.Wx {
		if g.Wx[i] != g2.Wx[i] {
			t.Errorf("Wx[%d] mismatch after round-trip", i)
			break
		}
	}
}

func TestGRUStep(t *testing.T) {
	t.Parallel()
	rng, _ := utils.NewRNG(42)
	g := NewGRU[float64](3, 4, 5)
	g.Init(rng)

	x := make([]float64, 4)
	h := g.Step(x)
	if len(h) != 5 {
		t.Errorf("Step output length = %d, want 5", len(h))
	}
	g.ResetState()
	h2 := g.Step(x)
	for i := range h {
		if h[i] != h2[i] {
			t.Errorf("Step after ResetState differs at index %d", i)
			break
		}
	}
}
