package params

import (
	"testing"

	"github.com/teratron/gonn/pkg"
)

func Test_checkActivationMode(t *testing.T) {
	tests := []struct {
		name string
		gave uint8
		want uint8
	}{
		{
			name: "#1_ModeTANH",
			gave: TANH,
			want: TANH,
		},
		{
			name: "#2_overflow",
			gave: 255,
			want: SIGMOID,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CheckActivationMode(tt.gave); got != tt.want {
				t.Errorf("checkActivationMode() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestActivation(t *testing.T) {
	type args struct {
		value pkg.FloatType
		mode  uint8
	}
	tests := []struct {
		name string
		args
		want pkg.FloatType
	}{
		{
			name: "#1_ModeLINEAR",
			args: args{.1, LINEAR},
			want: .1,
		},
		{
			name: "#2_ModeRELU",
			args: args{.1, RELU},
			want: .1,
		},
		{
			name: "#3_ModeRELU",
			args: args{-.1, RELU},
			want: 0,
		},
		{
			name: "#4_ModeLEAKYRELU",
			args: args{.1, LEAKYRELU},
			want: .1,
		},
		{
			name: "#5_ModeLEAKYRELU",
			args: args{-.1, LEAKYRELU},
			want: -.001,
		},
		{
			name: "#6_ModeSIGMOID",
			args: args{.1, SIGMOID},
			want: .52497918747894,
		},
		{
			name: "#7_ModeTANH",
			args: args{.1, TANH},
			want: .09966799462495583,
		},
		{
			name: "#8_default",
			args: args{.1, 255},
			want: .52497918747894,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Activation(tt.value, tt.mode); got != tt.want {
				t.Errorf("Activation() = %g, want %g", got, tt.want)
			}
		})
	}
}

func TestDerivative(t *testing.T) {
	type args struct {
		value pkg.FloatType
		mode  uint8
	}
	tests := []struct {
		name string
		args
		want pkg.FloatType
	}{
		{
			name: "#1_ModeLINEAR",
			args: args{.1, LINEAR},
			want: 1,
		},
		{
			name: "#2_ModeRELU",
			args: args{.1, RELU},
			want: 1,
		},
		{
			name: "#3_ModeRELU",
			args: args{-.1, RELU},
			want: 0,
		},
		{
			name: "#4_ModeLEAKYRELU",
			args: args{.1, LEAKYRELU},
			want: 1,
		},
		{
			name: "#5_ModeLEAKYRELU",
			args: args{-.1, LEAKYRELU},
			want: .01,
		},
		{
			name: "#6_ModeSIGMOID",
			args: args{.1, SIGMOID},
			want: .09000000000000001,
		},
		{
			name: "#7_ModeTANH",
			args: args{.1, TANH},
			want: .99,
		},
		{
			name: "#8_default",
			args: args{.1, 255},
			want: .09000000000000001,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Derivative(tt.value, tt.mode); got != tt.want {
				t.Errorf("Derivative() = %g, want %g", got, tt.want)
			}
		})
	}
}
