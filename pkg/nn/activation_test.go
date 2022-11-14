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
			name: "#1_LINEAR",
			args: args{.6, LINEAR},
			want: .6,
		},
		{
			name: "#2_RELU",
			args: args{-.6, RELU},
			want: 0,
		},
		{
			name: "#3_LEAKYRELU",
			args: args{.6, LEAKYRELU},
			want: .6,
		},
		{
			name: "#4_SIGMOID",
			args: args{.6, SIGMOID},
			want: .6456563,
		},
		{
			name: "#5_TANH",
			args: args{.6, TANH},
			want: .5370496,
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
		args args
		want float32
	}{
		{
			name: "#1_LINEAR",
			args: args{.6, LINEAR},
			want: 1,
		},
		{
			name: "#2_RELU",
			args: args{-.6, RELU},
			want: 0,
		},
		{
			name: "#3_LEAKYRELU",
			args: args{.6, LEAKYRELU},
			want: 1,
		},
		{
			name: "#4_SIGMOID",
			args: args{.6, SIGMOID},
			want: .24,
		},
		{
			name: "#5_TANH",
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
