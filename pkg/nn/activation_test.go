package nn

import "testing"

func TestActivation(t *testing.T) {
	type args struct {
		value float64
		mode  uint8
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Activation(tt.args.value, tt.args.mode); got != tt.want {
				t.Errorf("ActivationMode() = %v, want %v", got, tt.want)
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
		want float64
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Derivative(tt.args.value, tt.args.mode); got != tt.want {
				t.Errorf("Derivative() = %v, want %v", got, tt.want)
			}
		})
	}
}
