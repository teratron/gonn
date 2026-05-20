package transformer

import (
	"math"
	"testing"

	"github.com/teratron/gonn/pkg/activation"
)

func TestApplyActivationInPlace(t *testing.T) {
	cases := []struct {
		name   string
		mode   activation.Type
		input  []float64
		check  func(in, out float64) bool
	}{
		{
			name:  "ReLU zero clamp",
			mode:  activation.ReLU,
			input: []float64{-2, -1, 0, 1, 2},
			check: func(in, out float64) bool {
				want := math.Max(0, in)
				return math.Abs(out-want) < 1e-9
			},
		},
		{
			name:  "Linear identity",
			mode:  activation.Linear,
			input: []float64{-3, 0, 1.5, 42},
			check: func(in, out float64) bool {
				return math.Abs(out-in) < 1e-9
			},
		},
		{
			name:  "SIGMOID range",
			mode:  activation.SIGMOID,
			input: []float64{-10, 0, 10},
			check: func(in, out float64) bool {
				want := 1 / (1 + math.Exp(-in))
				return math.Abs(out-want) < 1e-6
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x := make([]float64, len(tc.input))
			copy(x, tc.input)
			applyActivationInPlace[float64](tc.mode, x)
			for i, v := range x {
				if !tc.check(tc.input[i], v) {
					// Use activation.Activation to get expected value for error message.
					want := activation.Activation(tc.input[i], tc.mode)
					t.Errorf("[%d] mode=%v in=%v: got %v want %v",
						i, tc.mode, tc.input[i], v, want)
				}
			}
		})
	}
}

// TestApplyActivationInPlace_InPlace confirms no new allocation — mutates x.
func TestApplyActivationInPlace_InPlace(t *testing.T) {
	x := []float64{-1, 2, -3, 4}
	ptr := &x[0]
	applyActivationInPlace[float64](activation.ReLU, x)
	if &x[0] != ptr {
		t.Error("applyActivationInPlace reallocated the slice")
	}
}

// TestApplyActivationInPlace_MatchesScalar verifies the slice-wise helper
// produces element-identical results to the scalar dispatcher for every mode.
func TestApplyActivationInPlace_MatchesScalar(t *testing.T) {
	modes := []activation.Type{
		activation.ReLU, activation.Linear, activation.SIGMOID, activation.TanH,
	}
	input := []float64{-2.5, -1, 0, 0.7, 3}
	for _, mode := range modes {
		x := make([]float64, len(input))
		copy(x, input)
		applyActivationInPlace[float64](mode, x)
		for i, v := range input {
			want := activation.Activation(v, mode)
			if math.Abs(x[i]-want) > 1e-9 {
				t.Errorf("mode=%v idx=%d: slice=%v scalar=%v", mode, i, x[i], want)
			}
		}
	}
}
