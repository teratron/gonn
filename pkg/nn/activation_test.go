package nn

import "testing"

func TestActivation(t *testing.T) {
	type args struct {
		value float64
		mode  uint8
	}
	tests := []struct {
		name string
		args
		want float32
	}{
		{
			name: "Activation_SIGMOID",
			args: args{.6, SIGMOID},
			want: .6456563,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := float32(Activation(tt.args.value, tt.args.mode)); got != tt.want {
				t.Errorf("Activation() = %v, want %v", got, tt.want)
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
		want float32
	}{
		{
			name: "Derivative_TANH",
			args: args{.6, TANH},
			want: .64,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := float32(Derivative(tt.args.value, tt.args.mode)); got != tt.want {
				t.Errorf("Derivative() = %v, want %v", got, tt.want)
			}
		})
	}
}
