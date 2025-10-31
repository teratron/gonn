package activation

import "math"

// TanH activation function: f(x) = tanh(x)
func tanhActivation[T float32 | float64](value T) T {
	return T(math.Tanh(float64(value)))
}

// TanH derivative function: f'(x) = 1 - tanh(x)^2
func tanhDerivative[T float32 | float64](value T) T {
	return T(1.0) - value*value
}
