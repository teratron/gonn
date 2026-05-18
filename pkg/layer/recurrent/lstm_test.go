package recurrent

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/utils"
)

func TestLSTMForward(t *testing.T) {
	t.Parallel()
	cases := []struct {
		seqLen, inSize, hidden int
		seed                   uint64
	}{
		{2, 3, 4, 2001},
		{5, 4, 8, 2002},
		{3, 2, 2, 2003},
	}
	for _, tc := range cases {
		rng, _ := utils.NewRNG(tc.seed)
		l := NewLSTM[float64](tc.seqLen, tc.inSize, tc.hidden)
		l.Init(rng)

		rng2, _ := utils.NewRNG(tc.seed + 1000)
		input := make([]float64, tc.seqLen*tc.inSize)
		for i := range input {
			input[i] = rng2.Float64()*2 - 1
		}

		out := l.Forward(input)
		if len(out) != tc.seqLen*tc.hidden {
			t.Errorf("Forward output length = %d, want %d", len(out), tc.seqLen*tc.hidden)
		}
		for i, v := range out {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("Forward output[%d] = %v (non-finite)", i, v)
			}
			// h_t = o*tanh(c_t), so |h_t| < 1
			if math.Abs(v) >= 1+1e-9 {
				t.Errorf("LSTM output[%d] = %v, must be in (-1, 1)", i, v)
			}
		}

		// Deterministic: same input → same output.
		out2 := l.Forward(input)
		for i := range out {
			if out[i] != out2[i] {
				t.Errorf("Forward is not deterministic at index %d", i)
				break
			}
		}
	}
}

func TestLSTMForgetBiasInit(t *testing.T) {
	t.Parallel()
	const hidden = 6
	rng, _ := utils.NewRNG(42)
	l := NewLSTM[float64](3, 4, hidden)
	l.Init(rng)

	// Forget-gate bias occupies B[hidden:2*hidden].
	for i := hidden; i < 2*hidden; i++ {
		if l.B[i] != 1.0 {
			t.Errorf("forget-gate bias B[%d] = %v, want 1.0", i, l.B[i])
		}
	}
	// All other gate biases must be zero.
	for i := range hidden {
		if l.B[i] != 0 {
			t.Errorf("input-gate bias B[%d] = %v, want 0", i, l.B[i])
		}
	}
	for i := 2 * hidden; i < 4*hidden; i++ {
		if l.B[i] != 0 {
			t.Errorf("cell/output-gate bias B[%d] = %v, want 0", i, l.B[i])
		}
	}
}

func TestLSTMRoundTrip(t *testing.T) {
	t.Parallel()
	rng, _ := utils.NewRNG(55)
	l := NewLSTM[float64](4, 3, 5)
	l.Init(rng)

	data, err := json.Marshal(l)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}

	var l2 LSTM[float64]
	if err := json.Unmarshal(data, &l2); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}

	if l2.SeqLen != l.SeqLen || l2.InSize != l.InSize || l2.Hidden != l.Hidden {
		t.Errorf("shape mismatch after round-trip")
	}
	for i := range l.Wx {
		if l.Wx[i] != l2.Wx[i] {
			t.Errorf("Wx[%d] mismatch: got %v, want %v", i, l2.Wx[i], l.Wx[i])
			break
		}
	}
	for i := range l.B {
		if l.B[i] != l2.B[i] {
			t.Errorf("B[%d] mismatch: got %v, want %v", i, l2.B[i], l.B[i])
			break
		}
	}
}

// TestLSTMGradient checks BPTT via finite differences (T-15T01).
func TestLSTMGradient(t *testing.T) {
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
		l := NewLSTM[float64](tc.seqLen, tc.inSize, tc.hidden)
		l.Init(rng)

		rng2, _ := utils.NewRNG(77)
		input := make([]float64, tc.seqLen*tc.inSize)
		upstream := make([]float64, tc.seqLen*tc.hidden)
		for i := range input {
			input[i] = rng2.NormFloat64() * 0.5
		}
		for i := range upstream {
			upstream[i] = rng2.NormFloat64() * 0.5
		}

		// Analytical gradient via Backward.
		l.Forward(input)
		gradX := l.Backward(upstream)

		// Numerical gradient via finite differences.
		for j := range input {
			orig := input[j]

			input[j] = orig + eps
			outPlus := l.Forward(input)
			lPlus := dot(outPlus, upstream)

			input[j] = orig - eps
			outMinus := l.Forward(input)
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
