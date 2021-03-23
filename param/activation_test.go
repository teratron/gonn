package param

import "testing"

func Test_checkActivationMode(t *testing.T) {
	tests := []struct {
		name string
		gave uint8
		want uint8
	}{
		{
			name: "#1_ModeTANH",
			gave: ModeTANH,
			want: ModeTANH,
		},
		{
			name: "#2_overflow",
			gave: 255,
			want: ModeSIGMOID,
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
		value float64
		mode  uint8
	}

	tests := []struct {
		name string
		args
		want float64
	}{
		{
			name: "#1_ModeLINEAR",
			args: args{.1, ModeLINEAR},
			want: .1,
		},
		{
			name: "#2_ModeRELU",
			args: args{.1, ModeRELU},
			want: .1,
		},
		{
			name: "#3_ModeRELU",
			args: args{-.1, ModeRELU},
			want: 0,
		},
		{
			name: "#4_ModeLEAKYRELU",
			args: args{.1, ModeLEAKYRELU},
			want: .1,
		},
		{
			name: "#5_ModeLEAKYRELU",
			args: args{-.1, ModeLEAKYRELU},
			want: -.001,
		},
		{
			name: "#6_ModeSIGMOID",
			args: args{.1, ModeSIGMOID},
			want: .52497918747894,
		},
		{
			name: "#7_ModeTANH",
			args: args{.1, ModeTANH},
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
		value float64
		mode  uint8
	}

	tests := []struct {
		name string
		args
		want float64
	}{
		{
			name: "#1_ModeLINEAR",
			args: args{.1, ModeLINEAR},
			want: 1,
		},
		{
			name: "#2_ModeRELU",
			args: args{.1, ModeRELU},
			want: 1,
		},
		{
			name: "#3_ModeRELU",
			args: args{-.1, ModeRELU},
			want: 0,
		},
		{
			name: "#4_ModeLEAKYRELU",
			args: args{.1, ModeLEAKYRELU},
			want: 1,
		},
		{
			name: "#5_ModeLEAKYRELU",
			args: args{-.1, ModeLEAKYRELU},
			want: .01,
		},
		{
			name: "#6_ModeSIGMOID",
			args: args{.1, ModeSIGMOID},
			want: .09000000000000001,
		},
		{
			name: "#7_ModeTANH",
			args: args{.1, ModeTANH},
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
