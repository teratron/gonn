package activation

import (
	"math"
	"testing"
)

func TestActivationFunctions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		mode     Type
		value    float64
		params   []float64
		expected float64
	}{
		// Test ELISH
		{"ELISH Positive", ELISH, 1.0, nil, 0.7310585786300049},  // Approximately 1 * sigmoid(1)
		{"ELISH Negative", ELISH, -1.0, nil, -0.170003401568548}, // (exp(-1)-1) * sigmoid(-1)

		// Test ELU
		{"ELU Positive", ELU, 1.0, nil, 1.0},
		{"ELU Negative", ELU, -1.0, nil, -0.632120558828577},                       // 1.0 * (exp(-1) - 1)
		{"ELU Negative with alpha", ELU, -1.0, []float64{2.0}, -1.264241117657115}, // 2.0 * (exp(-1) - 1)

		// Test LINEAR
		{"LINEAR Default", LINEAR, 5.0, nil, 5.0},
		{"LINEAR with slope", LINEAR, 5.0, []float64{2.0}, 10.0},
		{"LINEAR with slope and offset", LINEAR, 5.0, []float64{2.0, 3.0}, 13.0},

		// Test LEAKYRELU
		{"LEAKYRELU Positive", LEAKYRELU, 1.0, nil, 1.0},
		{"LEAKYRELU Negative", LEAKYRELU, -1.0, nil, -0.01}, // -1.0 * 0.01

		// Test RELU
		{"RELU Positive", RELU, 1.0, nil, 1.0},
		{"RELU Negative", RELU, -1.0, nil, 0.0},

		// Test SELU
		{"SELU Positive", SELU, 1.0, nil, 1.0507009873554805},  // 1.0507 * 1.0
		{"SELU Negative", SELU, -1.0, nil, -1.111330737812563}, // 1.0507 * 1.6733 * (exp(-1) - 1)

		// Test SIGMOID
		{"SIGMOID", SIGMOID, 0.0, nil, 0.5},
		{"SIGMOID Positive", SIGMOID, 1.0, nil, 0.731058578630049},
		{"SIGMOID Negative", SIGMOID, -1.0, nil, 0.2689414213699951},

		// Test SOFTMAX (simplified)
		{"SOFTMAX", SOFTMAX, 0.0, nil, 0.5},

		// Test SWISH
		{"SWISH", SWISH, 0.0, nil, 0.0},
		{"SWISH Positive", SWISH, 1.0, nil, 0.7310585786300049},

		// Test TANH
		{"TANH Zero", TANH, 0.0, nil, 0.0},
		{"TANH Positive", TANH, 1.0, nil, math.Tanh(1.0)},
		{"TANH Negative", TANH, -1.0, nil, math.Tanh(-1.0)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Activation(tt.value, tt.mode, tt.params...)
			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("Activation(%f, %v, %v) = %f, want %f", tt.value, tt.mode, tt.params, result, tt.expected)
			}
		})
	}
}

func TestDerivativeFunctions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		mode     Type
		value    float64
		params   []float64
		expected float64
	}{
		// Test ELISH derivative
		{"ELISH Derivative Positive", ELISH, 1.0, nil, 0.927670511871487},   // Calculated based on derivative formula
		{"ELISH Derivative Negative", ELISH, -1.0, nil, -0.025344425311521}, // Calculated based on derivative formula

		// Test ELU derivative
		{"ELU Derivative Positive", ELU, 1.0, nil, 1.0},
		{"ELU Derivative Negative", ELU, -1.0, nil, 0.36787944117144233}, // exp(-1)

		// Test LINEAR derivative
		{"LINEAR Derivative Default", LINEAR, 1.0, nil, 1.0},
		{"LINEAR Derivative with slope", LINEAR, 1.0, []float64{2.0}, 2.0},

		// Test LEAKYRELU derivative
		{"LEAKYRELU Derivative Positive", LEAKYRELU, 1.0, nil, 1.0},
		{"LEAKYRELU Derivative Negative", LEAKYRELU, -1.0, nil, 0.01},

		// Test RELU derivative
		{"RELU Derivative Positive", RELU, 1.0, nil, 1.0},
		{"RELU Derivative Negative", RELU, -1.0, nil, 0.0},

		// Test SELU derivative
		{"SELU Derivative Positive", SELU, 1.0, nil, 1.0507009873554805}, // scale = 1.0507
		{"SELU Derivative Negative", SELU, -1.0, nil, 0.646768603034812}, // scale * alpha * exp(-1)

		// Test SIGMOID derivative
		{"SIGMOID Derivative at 0", SIGMOID, 0.0, nil, 0.25},
		{"SIGMOID Derivative at 1", SIGMOID, 1.0, nil, 0.19661193324148185},

		// Test SOFTMAX derivative
		{"SOFTMAX Derivative", SOFTMAX, 0.0, nil, 0.25},

		// Test SWISH derivative
		{"SWISH Derivative at 0", SWISH, 0.0, nil, 0.5},

		// Test TANH derivative
		{"TANH Derivative at 0", TANH, 0.0, nil, 1.0}, // 1 - tanh(0)^2 = 1 - 0 = 1
		{"TANH Derivative at 1", TANH, 1.0, nil, 1.0 - math.Pow(math.Tanh(1.0), 2)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Derivative(tt.value, tt.mode, tt.params...)
			if math.Abs(result-tt.expected) > 1e-9 {
				t.Errorf("Derivative(%f, %v, %v) = %f, want %f", tt.value, tt.mode, tt.params, result, tt.expected)
			}
		})
	}
}

func TestEdgeCases(t *testing.T) {
	t.Parallel()

	// Test with float32
	result32 := Activation[float32](1.0, SIGMOID)
	expected32 := float32(0.7310585786300049)
	if math.Abs(float64(result32-float32(expected32))) > 1e-7 {
		t.Errorf("Activation[float32](1.0, SIGMOID) = %f, want %f", result32, expected32)
	}

	// Test with very large values
	resultLarge := Activation(100.0, SIGMOID)
	if resultLarge != 1.0 {
		t.Errorf("Activation(100.0, SIGMOID) should approach 1.0, got %f", resultLarge)
	}

	resultLargeNeg := Activation(-100.0, SIGMOID)
	if resultLargeNeg > 1e-10 { // Should be extremely close to 0
		t.Errorf("Activation(-100.0, SIGMOID) should approach 0.0, got %f", resultLargeNeg)
	}

	// Test derivatives with extreme values
	derivLarge := Derivative(100.0, SIGMOID)
	if derivLarge > 1e-10 {
		t.Errorf("Derivative(100.0, SIGMOID) should approach 0.0, got %f", derivLarge)
	}
}

func BenchmarkActivation(b *testing.B) {
	b.Run("Sigmoid", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Activation(1.0, SIGMOID)
		}
	})

	b.Run("Tanh", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Activation(1.0, TANH)
		}
	})

	b.Run("ReLU", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Activation(1.0, RELU)
		}
	})
}

func BenchmarkDerivative(b *testing.B) {
	b.Run("Sigmoid", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Derivative(1.0, SIGMOID)
		}
	})

	b.Run("Tanh", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Derivative(1.0, TANH)
		}
	})

	b.Run("ReLU", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Derivative(1.0, RELU)
		}
	})
}
