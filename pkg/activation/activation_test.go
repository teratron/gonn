package activation

import (
	"testing"
)

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
