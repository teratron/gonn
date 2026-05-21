package activation

import (
	"math"
	"testing"
)

// TestSoftmaxActivationStability verifies that softmaxActivation never returns NaN
// or Inf for extreme positive and negative inputs. The pre-Phase-fix implementation
// computed exp(x)/(exp(x)+1) which overflows to NaN when x > ~88 (float32 limit).
func TestSoftmaxActivationStability(t *testing.T) {
	probes := []float64{0, 1, -1, 88, 89, 100, 500, 1000, -88, -100, -500}
	for _, v := range probes {
		got := softmaxActivation(v)
		if math.IsNaN(float64(got)) {
			t.Errorf("softmaxActivation(%v) = NaN", v)
		}
		if math.IsInf(float64(got), 0) {
			t.Errorf("softmaxActivation(%v) = Inf", v)
		}
		if got < 0 || got > 1 {
			t.Errorf("softmaxActivation(%v) = %v, want in [0,1]", v, got)
		}
	}
}

func TestActivationTypeString(t *testing.T) {
	tests := []struct {
		mode Type
		want string
	}{
		{ELISH, "ELISH"},
		{ELU, "ELU"},
		{Linear, "Linear"},
		{LeakyReLU, "LeakyReLU"},
		{ReLU, "ReLU"},
		{SELU, "SELU"},
		{SIGMOID, "SIGMOID"},
		{SOFTMAX, "SOFTMAX"},
		{SWISH, "SWISH"},
		{TanH, "TanH"},
		{Type(99), "Unknown"}, // Test unknown value
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.mode.String(); got != tt.want {
				t.Errorf("Type.String() = %v, want %v", got, tt.want)
			}
		})
	}
}
